! Created by Michael Ou
! SPARKS — Sequential Pilot-point Assisted Random-path Kriging and Simulation
program sparks
  use ieee_arithmetic
  use iso_fortran_env, only: input_unit, error_unit, output_unit
  use io
  use f90getopt
  use common
  use rotation
  use variogram
  use kdtree2_module
  use gaussian_quadrature
  use kriging                  ! provides t_kriging; replaces hand-rolled solve machinery
  use kriging_err, only: kriging_failed, kriging_last_error
  implicit none

  character(8), parameter           :: sparks_version = '20260515'
#ifndef GIT_HASH
#  define GIT_HASH   "n/a"
#endif
#ifndef FC_NAME
#  define FC_NAME    "unknown"
#endif
#ifndef FC_VERSION
#  define FC_VERSION "unknown"
#endif
  character(*), parameter :: sparks_githash  = GIT_HASH
  character(*), parameter :: sparks_compiler = FC_NAME
  character(*), parameter :: sparks_fc_ver   = FC_VERSION
  ! ---------------------------------------------------------------------------
  ! t_kriging encapsulates: obs, grid, blocks, vgm, search, solve, estimate.
  ! The program's job is now:
  !   1. Parse CLI args
  !   2. Read data files into plain arrays
  !   3. Transfer data into krig via set_vgm / set_obs / set_grid / set_sim / set_search
  !   4. Call krig%solve()
  !   5. Write krig%block%estimate to output
  ! ---------------------------------------------------------------------------
  type(t_kriging)                   :: krig

  type(option_s), allocatable       :: opts(:)
  character(2048) :: obsfile1, obsfile2, gridfile, facfile, blockfile, outfile, randpath, samfile, fomt, rsfile, lnfile

  ! inputs — scalars
  integer         :: ndim, ndrift, nsim, block_type, seed, unbias, ngrid
  integer         :: nobs1, nmax1
  integer         :: nobs2, nmax2
  integer         :: nblock
  integer         :: nthread
  real            :: sk_mean, maxdist1, maxdist2, vmin, vmax, blocksize(3)

  ! inputs — allocatable arrays
  integer, allocatable :: irandpath(:), nblockpnt(:)          ! blocks-kriging geometry
  real, allocatable :: obs1(:, :), obs1drift(:, :)   ! (xyz+value[+variance], nobs1)
  real, allocatable :: obs2(:, :), obs2drift(:, :)   ! (xyz+value[+variance], nobs2)
  real, allocatable :: grid(:, :)    ! (xyz[+weight], ngrid)
  real, allocatable :: blocks(:, :), blockdrift(:,:)  ! (ndim, nblock)
  real, allocatable :: rangescale(:), localnugget(:)
  real, allocatable :: samples(:, :)                           ! (nsim, nblock) — passed to set_sim

  ! raw 4-token CLI variogram specs: "type range sill nugget"
  character(256)  :: vgm_spec1(99), vgm_spec2(99), vgm_specc(99)
  ! rotation/anisotropy and flags
  real            :: ang1, ang2, ang3, anis1, anis2
  logical         :: correct_weight, writexy, neglect_error, writemat, showargs
  logical         :: loocv, anisotropic_search, verbose, blockpntweight, obserror

  ! local bookkeeping
  character(len=2)  :: opt
  real, parameter   :: large = 3.4028235E+38
  logical           :: use_old_weight, store_weight
  integer           :: ifile, ioerr, ifilefac, i, ii, jj, kk, ig, iv, isim, iout, ib, ntp
  integer           :: nvar = 1
  integer           :: nvgm1 = 0, nvgm2 = 0, nvgmc = 0

  ! namelist input file path (set by -nl / --namelist)
  character(2048) :: nmlfile

  ! ---------------------------------------------------------------------------
  ! Namelist groups — variable names match the program variables exactly so
  ! that read(ifile, nml=<group>) populates them directly.
  ! ---------------------------------------------------------------------------
  namelist /input_output/ &
    obsfile1, obsfile2, gridfile, facfile, blockfile, randpath, samfile, rsfile, lnfile, fomt, outfile

  namelist /dims/ &
    ndim, ndrift, nobs1, nobs2, ngrid, nblock, nsim

  namelist /krige_opt/ &
    nmax1, nmax2, unbias, maxdist1, maxdist2, sk_mean, vmin, vmax, block_type, blocksize, seed, nthread

  namelist /variograms/ &
    vgm_spec1, vgm_spec2, vgm_specc

  namelist /anisotropy/ &
    ang1, ang2, ang3, anis1, anis2

  namelist /flags/ &
    correct_weight, writexy, neglect_error, writemat, &
    showargs, loocv, anisotropic_search, verbose, blockpntweight, obserror

  allocate (opts, source=(/ &
    option_s("namelist",        "nl", 1, "read all settings from a namelist (.nml) file. all other options are ignored if namelist is set."), &
    option_s("dim",             "d",  5, "define spatial dimensions: ndim nobs1 nblock nobs2 ndrift."), &
    option_s("obsfile1",        "of", 1, "primary observation file. columns: id,x(,y,z),value[,variance][,drifts]."), &
    option_s("facfile",         "ff", 1, "interpolation factor file. created when both facfile and blockfile are defined."), &
    option_s("blockfile",       "bf", 1, "blocks file. columns: id,x(,y,z)[,drifts]. if gridfile is defined, last column gives number of grid points per blocks."), &
    option_s("gridfile",        "gf", 1, "optional grid file for blocks kriging. columns: id,x(,y,z)[,weight]."), &
    option_s("blockpntweight",  "bw", 0, "read point weights for blocks kriging from the grid file. default: equal weights."), &
    option_s("pathfile",        "pf", 1, "file of simulation path indices. if omitted a random path is generated."), &
    option_s("samfile",         "sf", 1, "file of pre-generated standard-normal samples."), &
    option_s("obsfile2",        "o2", 1, "secondary (covariate) observation file. columns: id,x(,y,z),value[,drifts]."), &
    option_s("seed",            "sd", 1, "seed for random path and sample generation."), &
    option_s("sim",             "s",  0, "enable sequential gaussian simulation. default: disabled."), &
    option_s("nmax1",           "n1", 1, "max primary observations per neighbourhood. default: 200."), &
    option_s("nmax2",           "n2", 1, "max covariate observations per neighbourhood. default: 200."), &
    option_s("rangescale",      "rs", 1, "file of per-blocks variogram range scale factors."), &
    option_s("localnugget",     "ln", 1, "file of per-blocks local nugget values."), &
    option_s("skmean",          "sm", 1, "mean for simple kriging. default: 0."), &
    option_s("unbias",          "u",  0, "unbiasedness constraint: off=simple kriging, on=ordinary kriging."), &
    option_s("ang1",            "a1", 1, "azimuth angle of principal direction (degrees). default: 0."), &
    option_s("ang2",            "a2", 1, "dip angle of principal direction (degrees). default: 0."), &
    option_s("ang3",            "a3", 1, "third rotation / plunge angle (degrees). default: 0."), &
    option_s("anis1",           "s1", 1, "first anisotropy ratio (minor/major range). default: 1."), &
    option_s("anis2",           "s2", 1, "second anisotropy ratio (minor/major range). default: 1."), &
    option_s("vario1",          "v1", 4, "primary variogram: type range sill nugget."), &
    option_s("vario2",          "v2", 4, "secondary variogram. same format as vario1."), &
    option_s("varioc",          "vc", 4, "cross-variogram (primary x secondary). same format as vario1."), &
    option_s("bounds",          "bd", 2, "lower and upper bounds for simulated/estimated values."), &
    option_s("blocksize",       "bs", 1, "blocks dimensions for gaussian quadrature (block_type=-4)."), &
    option_s("maxdist",         "md", 2, "max search distance: two values for primary and covariate."), &
    option_s("correct",         "cw", 0, "remove negative weights and renormalise to sum to 1."), &
    option_s("anisosearch",     "as", 0, "neighbour search in rotated/scaled (anisotropic) coordinates."), &
    option_s("obserror",        "oe", 0, "observation error variance is present in the input file."), &
    option_s("fmt",             "fm", 1, "fortran write format for output. default: '(G0.12)'."), &
    option_s("writexy",         "xy", 0, "include id and coordinates in output. default: values only."), &
    option_s("continue",        "cc", 0, "set failed blocks to NaN instead of aborting."), &
    option_s("verbose",         "v",  0, "print progress messages during solving."), &
    option_s("nthread",         "nt", 1, "number of OpenMP threads for the solve loop. 0 (default) uses the OMP runtime setting (OMP_NUM_THREADS)."), &
    option_s("loocv",           "cv", 0, "leave-one-out cross-validation: grid equals observations."), &
    option_s("writemat",        "wm", 0, "write kriging matrix and rhs to files for debugging."), &
    option_s("showargs",        "sa", 0, "print full configuration summary before solving."), &
    option_s("help",            "h",  0, "show this help message.") &
  /))

  ! ---- default values -------------------------------------------------------
  fomt = '(10(G0.12,:x))'
  outfile = '~'
  obsfile1 = ''; obsfile2 = ''; gridfile = ''; facfile = ''
  randpath = ''; samfile = ''; blockfile = ''; rsfile = ''; lnfile=''
  maxdist1 = large; maxdist2 = large
  nmax1 = 0; nmax2 = 0
  ndrift = 0; unbias = 0; seed = 0
  ang1 = zero; ang2 = zero; ang3 = zero
  anis1 = one; anis2 = one
  nsim = 0; block_type = 0; nthread = 0
  correct_weight = .false.
  ndim = 2; nobs1 = 0; nobs2 = 0; ngrid = 0; nblock = 0; nvar = 1
  vmin = -large; vmax = large
  writexy = .false.; verbose = .false.; neglect_error = .false.
  writemat = .false.; showargs = .false.; loocv = .false.
  anisotropic_search = .false.; blockpntweight = .false.; obserror = .false.
  blocksize = zero
  vgm_spec1 = ''; vgm_spec2 = ''; vgm_specc = ''

  ! ---- parse CLI ------------------------------------------------------------
  ! Pass 1: check for -nl / --namelist before anything else.
  ! If found, read all settings from the namelist file and skip CLI parsing.
  nmlfile = ''
  do
    opt = getopt(opts)
    if (trim(opt) == 'nl') then
      nmlfile = adjustl(trim(optarg))
      exit
    end if
    if (trim(opt) == char(0)) exit
  end do
  call reset_opt()

  if (nmlfile /= '') then
    ! Namelist path: read all settings from file; skip CLI entirely.
    call read_namelist(nmlfile)
  else
    ! CLI path: pass 2 — set verbose early so subsequent output is visible.
    do
      opt = getopt(opts)
      if (trim(opt) == 'v') then; verbose = .true.; exit; end if
      if (trim(opt) == char(0)) exit
    end do
    call reset_opt()

    ! CLI path: pass 3 — parse all remaining options.
    ndrift = -1   ! sentinel: must be set via -d
    do
      opt = getopt(opts)
      select case (trim(opt))
      case (char(0))
        if (len_trim(optarg) > 0) outfile = adjustl(trim(optarg))
        exit

      case ("d")
        read (optarg, *) ndim, nobs1, nblock, nobs2, ndrift
        do i = 1, size(opts)
          if (opts(i)%short == "bs") opts(i)%narg = ndim
        end do

      case ("of"); obsfile1 = adjustl(optarg)
      case ("o2"); obsfile2 = adjustl(optarg)
      case ("gf"); gridfile = adjustl(optarg)
      case ("bf"); blockfile = adjustl(optarg)
      case ("rs"); rsfile = adjustl(optarg)
      case ("ln"); lnfile = adjustl(optarg)
      case ("u"); unbias = 1
      case ("bw"); blockpntweight = .true.
      case ("cw"); correct_weight = .true.
      case ("ff"); facfile = adjustl(optarg)
      case ("pf"); randpath = adjustl(optarg)
      case ("sf"); samfile = adjustl(optarg)
      case ("sd"); read (optarg, *) seed
      case ("s"); nsim = 1
      case ("sm"); read (optarg, *) sk_mean
      case ("n1"); read (optarg, *) nmax1
      case ("n2"); read (optarg, *) nmax2
      case ("a1"); read (optarg, *) ang1
      case ("a2"); read (optarg, *) ang2
      case ("a3"); read (optarg, *) ang3
      case ("s1"); read (optarg, *) anis1
      case ("s2"); read (optarg, *) anis2
      case ("bd"); read (optarg, *) vmin, vmax
      case ("bs"); read (optarg, *) blocksize(:ndim)
      case ("md"); read (optarg, *) maxdist1, maxdist2

        ! Variogram specs: store CLI strings; parsed and forwarded by set_vgm_().
        ! CLI format: "type range sill nugget"
      case ("v1")
        nvgm1 = nvgm1 + 1
        vgm_spec1(nvgm1) = adjustl(trim(optarg))
      case ("vc")
        nvgmc = nvgmc + 1
        vgm_specc(nvgmc) = adjustl(trim(optarg))
      case ("v2")
        nvgm2 = nvgm2 + 1
        vgm_spec2(nvgm2) = adjustl(trim(optarg))

      case ("fm"); read (optarg, *) fomt
      case ("oe"); obserror = .true.
      case ("cc"); neglect_error = .true.
      case ("nt"); read (optarg, *) nthread
      case ("xy"); writexy = .true.
      case ("v"); verbose = .true.
      case ("cv"); loocv = .true.
      case ("as"); anisotropic_search = .true.
      case ("wm"); writemat = .true.
      case ("sa"); showargs = .true.
      case ("h"); call showhelp()
      case default; call perr("  Error: unknown command-line option.")
      end select
      ! Safety exit: if -d was never parsed, ndrift stays at its sentinel value (-1).
      ! The sanity check below will report the error; no need to keep parsing.
      if (ndrift == -1) exit
    end do
  end if   ! nmlfile == '' (CLI path)
  if (verbose) call print_banner()
  ! ---- sanity checks -----------------------------------------------

  if (ndrift < 0) call perr("  Error: ndrift not set. Use -d or define &dims in the namelist.")

  ! --- dimensions
  if (ndim < 1 .or. ndim > 3) &
    call perr("  Error: ndim must be 1, 2, or 3.")
  if (nobs1 <= 0) &
    call perr("  Error: nobs1 must be > 0.")
  if (nblock <= 0 .and. .not. loocv) &
    call perr("  Error: nblock must be > 0 (set via -d or &dims).")

  ! --- files
  if (len_trim(obsfile1) == 0) &
    call perr("  Error: primary observation file not specified (-of / obsfile1).")
  if (nobs2 > 0 .and. len_trim(obsfile2) == 0) &
    call perr("  Error: nobs2 > 0 but no secondary observation file specified (-o2).")
  if (.not. loocv .and. blockfile == "" .and. facfile == "") &
    call perr("  Error: either blockfile (-bf) or facfile (-ff) must be specified.")

  ! --- variograms
  if (nvgm1 == 0) &
    call perr("  Error: primary variogram not set (-v1 / vgm_spec1).")
  if (nobs2 > 0) then
    nvar = 2
    if (nvgm2 == 0 .or. nvgmc == 0) &
      call perr("  Error: variogram not set for covariate (-v2) or cross-variogram (-vc).")
  end if

  ! --- block / grid geometry
  if (.not. loocv) then
    if (any(blocksize > zero) .and. gridfile /= '') &
      call perr("  Error: blocksize (block_type=-4) and gridfile (block_type>0) are mutually exclusive.")
    if (blockpntweight .and. gridfile == '') &
      call perr("  Error: -bw (blockpntweight) requires a gridfile (-gf).")
    if (block_type > 0 .and. len_trim(blockfile) == 0) &
      call perr("  Error: block_type > 0 requires blockfile (-bf).")
    if (block_type == -4 .and. all(blocksize == zero)) &
      call perr("  Error: block_type=-4 (Gaussian quadrature) requires blocksize (-bs).")
  end if

  ! --- anisotropy
  if (anis1 <= zero .or. anis1 > one) &
    call perr("  Error: anis1 must satisfy 0 < anis1 <= 1.")
  if (anis2 <= zero .or. anis2 > one) &
    call perr("  Error: anis2 must satisfy 0 < anis2 <= 1.")

  ! --- bounds
  if (vmin >= vmax) &
    call perr("  Error: vmin must be strictly less than vmax.")

  ! --- simulation
! SGS reminder: variogram sill should equal 1 for correct simulation variance.
! This is not enforced here because nested structures may sum to 1 individually.
  if (nsim > 0) then
    if (verbose) print "(A)", ' SGSIM activated. Note: SGSIM recomands standardised variogram (total sill = 1.0).'
    if (nmax1==0) call perr("  Error: nmax1 must be > 0 for simulation.")
    if (loocv) call perr("  Error: simulation cannot be used with leave-one-out cross-validation.")
  end if
  if (loocv) nblock = nobs1

  ! --- factor file
  ! facfile + no blockfile       → load previously stored weights from facfile; skip solve
  use_old_weight = facfile /= '' .and. blockfile == ''
  ! facfile + blockfile present  → solve normally and store weights to facfile for reuse
  store_weight = facfile /= '' .and. blockfile /= ''

  ! ---- initialize t_kriging -----------------------------------------------
  call krig%initialize( &
    ndim=ndim, &
    nvar=nvar, &
    ndrift=max(ndrift, 0), &
    unbias=unbias, &
    nsim=nsim, &
    anisotropic_search=anisotropic_search, &
    weight_correction=correct_weight, &
    cross_validation=loocv, &
    write_mat=writemat, &
    neglect_error=neglect_error, &
    verbose=verbose, &
    weight_file=facfile, &
    store_weight=store_weight, &
    use_old_weight=use_old_weight, &
    bounds=[vmin, vmax], &
    seed=seed)
  call stop_if_kriging_failed('initializing kriging')

  ! ---- allocations ----------------------------------------------------
  ! Allocate all major arrays up-front using dimension scalars already known
  ! from CLI / namelist, before any file reads.
  !
  ! block_type encoding:
  !   -4  Gaussian quadrature; integration points derived from blocksize (-bs)
  !    0  point kriging; each row of blockfile is a single target point
  !   >0  block kriging with explicit discretization points supplied in gridfile
  if (blocksize(1) > zero) then
    block_type = -4
    ngrid = nblock
  else if (gridfile /= '') then
    ngrid = linecount(gridfile) - 1   ! header line excluded
    block_type = 1
  else
    block_type = 0
    ngrid = nblock
  end if
  ! ntp: column count in the obs array — coordinates + value + (optional) variance
  ntp = ndim + 1
  if (obserror) ntp = ntp + 1
  allocate (obs1(ntp, nobs1))
  if (nobs2 > 0) then
    allocate (obs2(ntp, nobs2))
  end if
  if (ndrift > 0) then
    allocate (obs1drift(ndrift, nobs1))
    allocate (blockdrift(ndrift, nblock))
    if (nobs2 > 0) then
      allocate (obs2drift(ndrift, nobs2))
    end if
  end if
  if (.not. loocv) then
    allocate (blocks(ndim, nblock))
    ! ntp reused: column count in the grid array — coordinates + (optional) point weight
    ntp = ndim
    if (blockpntweight) ntp = ntp + 1
    allocate (grid(ntp, ngrid))
  end if
  allocate (irandpath(nblock)); irandpath = [(ib, ib=1, nblock)]   ! default: sequential path
  allocate (rangescale(nblock)); rangescale = 1.0    ! default: no range scaling
  allocate (localnugget(nblock)); localnugget = 0.0  ! default: no local nugget
  if (nsim > 0) allocate (samples(nsim, nblock))

  call set_vgm_()

  ! ---- read obs ------------------------------------------------------
  call set_obs_(1, obsfile1, nobs1, nmax1, maxdist1, obs1, obs1drift)
  if (nobs2 > 0) call set_obs_(2, obsfile2, nobs2, nmax2, maxdist2, obs2, obs2drift)

  ! ---- read grid ------------------------------------------------------
  call set_grid_()

  ! ---- prepare ------------------------------------------------------
  if (nsim > 0) call set_sim_()
  call set_search_()

  ! ---- show options after krig setup -------------------
  if (showargs) call showoptions()
  ! ---- hand data to t_kriging and solve -------------------------------------
  call krig%solve(nthread)
  call stop_if_kriging_failed('solving kriging system')
  call write_output()
  call krig%finalize()
  if (verbose) print *, "SPARKS exited peacefully."

contains

  ! ===========================================================================
  ! read_namelist
  !
  ! Reads all six namelist groups from a single .nml file.  Each group is
  ! optional — a missing group leaves the corresponding variables at their
  ! default values.  The file is rewound between groups because each
  ! read(nml=) stops immediately after the closing '/'.
  ! ===========================================================================
  subroutine read_namelist(afile)
    character(*), intent(in) :: afile
    integer :: istat

    if (verbose) print *, 'Reading namelist from "'//trim(afile)//'"'
    open (newunit=ifile, file=trim(afile), status='old', iostat=istat)
    if (istat /= 0) call perr('  Error: cannot open namelist file "'//trim(afile)//'"')

    read (ifile, nml=input_output, iostat=istat); if (istat > 0) call nml_warn('input_output', istat); rewind (ifile)
    read (ifile, nml=dims, iostat=istat); if (istat > 0) call nml_warn('dims', istat); rewind (ifile)
    read (ifile, nml=krige_opt, iostat=istat); if (istat > 0) call nml_warn('krige_opt', istat); rewind (ifile)
    read (ifile, nml=variograms, iostat=istat); if (istat > 0) call nml_warn('variograms', istat); rewind (ifile)
    read (ifile, nml=anisotropy, iostat=istat); if (istat > 0) call nml_warn('anisotropy', istat); rewind (ifile)
    read (ifile, nml=flags, iostat=istat); if (istat > 0) call nml_warn('flags', istat)

    close (ifile)
    ! Count non-empty variogram specs from namelist
    if (nvgm1 == 0) nvgm1 = count(len_trim(vgm_spec1) > 0)
    if (nvgm2 == 0) nvgm2 = count(len_trim(vgm_spec2) > 0)
    if (nvgmc == 0) nvgmc = count(len_trim(vgm_specc) > 0)
    if (verbose) print *, 'Namelist read successfully.'
  end subroutine read_namelist

  subroutine nml_warn(group, istat)
    use iso_fortran_env, only: iostat_end
    character(*), intent(in) :: group
    integer, intent(in) :: istat
    ! iostat_end means the group was simply absent — that is fine.
    if (istat /= iostat_end) then
      write (error_unit, '(A,I0)') &
        '  Warning: error reading namelist group &'//group//', iostat=', istat
    end if
  end subroutine nml_warn

  ! ===========================================================================
  ! Data reading helpers
  ! ===========================================================================

  ! Parse a 4-token CLI variogram spec ("type range sill nugget") and register
  ! it with krig%set_vgm.  Minor semi-axes come from the global anis1/anis2
  ! ratios; rotation angles from ang1/ang2/ang3.
  subroutine apply_vgm_(ivar, jvar, spec)
    integer,      intent(in) :: ivar, jvar
    character(*), intent(in) :: spec
    character(24) :: vtype
    real          :: nugget, sill, a_major, a_minor1, a_minor2

    read(spec, *) vtype, a_major, sill, nugget
    a_minor1 = a_major * anis1
    a_minor2 = a_major * anis2
    call krig%set_vgm(ivar=ivar, jvar=jvar, vtype=trim(vtype), &
                      nugget=nugget, sill=sill, &
                      a_major=a_major, a_minor1=a_minor1, a_minor2=a_minor2, &
                      azimuth=ang1, dip=ang2, plunge=ang3)
    call stop_if_kriging_failed('setting variogram')
  end subroutine apply_vgm_

  subroutine set_vgm_()
    ! Registration order matters: primary auto (1,1), cross (1,2), secondary auto (2,2).
    if (verbose) print *, 'Setting variograms'
    do ii = 1, nvgm1
      call apply_vgm_(1, 1, vgm_spec1(ii))
    end do
    do ii = 1, nvgmc
      call apply_vgm_(1, 2, vgm_specc(ii))   ! cross-variogram
    end do
    do ii = 1, nvgm2
      call apply_vgm_(2, 2, vgm_spec2(ii))
    end do
  end subroutine set_vgm_

  subroutine set_obs_(ivar, obsfile, nobs, nmax, maxdist, obs, obsdrift)
    integer, intent(in) :: ivar, nobs, nmax
    real, intent(in) :: maxdist
    character(len=*), intent(in) :: obsfile
    real, intent(out) :: obs(:, :)
    real, intent(out) :: obsdrift(:, :)

    if (verbose) print "(A,I0,A)", ' Reading OBS', ivar, ' in "'//trim(obsfile)//'"'
    if (ndrift > 0) then
      call read_data(obsfile, nobs, ndim+1, ndrift, obs, obsdrift, ioerr)
    else
      call read_data(obsfile, nobs, ndim+1, obs, ioerr)
    end if
    if (obserror) then
   call krig%set_obs(ivar=ivar, coord=obs(1:ndim, :), value=obs(ndim + 1, :), variance=obs(ndim + 2, :), nmax=nmax, maxdist=maxdist, sk_mean=sk_mean)
    else
      call krig%set_obs(ivar=ivar, coord=obs(1:ndim, :), value=obs(ndim + 1, :), nmax=nmax, maxdist=maxdist, sk_mean=sk_mean)
    end if
    call stop_if_kriging_failed('setting observations')
    if (ndrift > 0) then
      call krig%set_obs_drift(ivar, obsdrift)
      call stop_if_kriging_failed('setting observation drift')
    end if
  end subroutine set_obs_

  subroutine set_grid_()
  if (loocv) then
    ! LOOCV mode, grid is not needed
    nblock = nobs1
    call krig%set_grid()
    call stop_if_kriging_failed('setting LOOCV grid')
  else
    if (verbose) print *, 'Reading BLOCK in "'//trim(blockfile)//'"'
    if (ndrift > 0) then
      if (block_type > 0) then
        call read_data(blockfile, nblock, ndim, ndrift, blocks, blockdrift, nblockpnt, ioerr)
      else
        call read_data(blockfile, nblock, ndim, ndrift, blocks, blockdrift, ioerr)
      end if
    else
      if (block_type > 0) then
        call read_data(blockfile, nblock, ndim, blocks, nblockpnt, ioerr)
      else
        call read_data(blockfile, nblock, ndim, blocks, ioerr)
      end if
    end if
    if (rsfile /= '') then
      if (verbose) print *, 'Reading RANGESCALE in "'//trim(rsfile)//'"'
      call read_data(rsfile, rangescale, ioerr)
    end if
    if (lnfile /= '') then
      if (verbose) print *, 'Reading LOCALNUGGET in "'//trim(lnfile)//'"'
      call read_data(lnfile, localnugget, ioerr)
    end if
    if (block_type > 0) then
      ! Block kriging: read explicit discretization points from gridfile.
      ! nblockpnt(ib) contains the number of grid points belonging to block ib.
      ntp = ndim
      if (blockpntweight) ntp = ndim + 1
      call read_data(gridfile, ngrid, ntp, grid, ioerr)
      if (blockpntweight) then
        call krig%set_grid(grid(:ndim,:), block_type, nblockpnt=nblockpnt, pointweight=grid(ndim+1,:), rangescale=rangescale, localnugget=localnugget)
      else
        call krig%set_grid(grid(:ndim, :), block_type, nblockpnt=nblockpnt, rangescale=rangescale, localnugget=localnugget)
      end if
    else if (block_type == 0) then
      ! Point kriging: each block row is a single target point.
      call krig%set_grid(blocks(:ndim, :), block_type, rangescale=rangescale, localnugget=localnugget)
    else
      ! Gaussian quadrature (block_type = -4): integration points are generated
      ! internally from the block centres in `blocks` and the supplied blocksize.
      ! spread() broadcasts the 1-D blocksize vector into a (ndim, nblock) matrix.
      call krig%set_grid(blocks(:ndim,:), block_type, blocksize=spread(blocksize, 2, nblock), rangescale=rangescale, localnugget=localnugget)
    end if
    call stop_if_kriging_failed('setting grid')
    if (ndrift > 0) then
      call krig%set_grid_drift(blockdrift, ivar=-1) ! broadcast to all variables
      call stop_if_kriging_failed('setting grid drift')
    end if
  end if
  end subroutine set_grid_

  subroutine set_sim_()
    ! Four combinations of samfile / randpath availability:
    !   samfile + randpath → use supplied samples and supplied path
    !   samfile only       → use supplied samples; krig generates the path
    !   randpath only      → use supplied path; krig generates samples
    !   neither            → krig generates both samples and path from seed
    real :: samples_tmp(nsim, nvar, nblock)
    if (samfile /= '') then
      if (verbose) print *, 'Reading SAMPLE in "'//trim(samfile)//'"'
      call read_data(samfile, samples, ioerr)
      do ii = 1, nvar
        samples_tmp(:, ii, :) = samples
      end do
      if (randpath /= '') then
        if (verbose) print *, 'Reading PATH in "'//trim(randpath)//'"'
        call read_data(randpath, nblock, irandpath, ioerr)
        call krig%set_sim(irandpath, samples_tmp)
      else
        call krig%set_sim(sample=samples_tmp)
      end if
    else
      if (randpath /= '') then
        if (verbose) print *, 'Reading PATH in "'//trim(randpath)//'"'
        call read_data(randpath, nblock, irandpath, ioerr)
        call krig%set_sim(irandpath)
      else
        call krig%set_sim()
      end if
    end if
    call stop_if_kriging_failed('setting simulation inputs')
  end subroutine set_sim_

  subroutine set_search_()
    call krig%set_search(1, anis1, anis2, ang1, ang2, ang3)
    call stop_if_kriging_failed('setting primary search')
    if (nobs2 > 0) then
      call krig%set_search(2, anis1, anis2, ang1, ang2, ang3)
      call stop_if_kriging_failed('setting secondary search')
    end if
  end subroutine set_search_

  ! ===========================================================================
  ! Output helpers
  ! ===========================================================================

  subroutine open_output()
    character :: cname(3) = ['x', 'y', 'z']
    character(15)  :: strsim(nsim + 1)
    if (trim(outfile) == '~') then
      iout = output_unit
    else
      open (newunit=iout, file=trim(outfile), status='replace')
    end if
    if (writexy) then
      do ii = 1, nsim
        write (strsim(ii), '(A,I0)') 'estimate', ii
      end do
      if (nsim == 0) then
        if (loocv) then
          write (iout, '(99(A,:,","))') 'igrid', cname(1:ndim), 'observed', 'estimate', 'variance'
        else
          write (iout, '(99(A,:,","))') 'igrid', cname(1:ndim), 'estimate', 'variance'
        end if
      else
        write (iout, '(99(A,:,","))') 'igrid', cname(1:ndim), (trim(strsim(ii)), ii=1, nsim), 'variance'
      end if
    end if
  end subroutine open_output

  subroutine write_output()
    integer :: ib2
    ! Extract results from t_kriging%block
    call open_output()
    if (writexy) then
      do ib2 = 1, krig%block%n
        if (loocv) then
          write (iout, "(I0,*(:,',',G0.12))") &
            ib2, krig%block%coord(:, ib2), krig%obs(1)%value(1,1,ib2), &
            krig%block%value(1, 1, ib2), krig%block%variance(1, 1, ib2)
        else
          write (iout, "(I0,*(:,',',G0.12))") &
            ib2, krig%block%coord(:, ib2), krig%block%value(:, 1, ib2), krig%block%variance(1, 1, ib2)
        end if
      end do
    else
      do ib2 = 1, krig%block%n
        write (iout, "(*(G0.12,:,','))") krig%block%value(:, 1, ib2)
      end do
    end if

    if (iout /= output_unit) close (iout)
    if (verbose) print *, 'Results have been written successfully'
  end subroutine write_output

  subroutine perr(msg)
    character(*) :: msg
    write (error_unit, '(A)') msg
    ! SPARKS is an executable, so STOP exits only this CLI process.  Library
    ! code still records errors through kriging_err so Python callers survive.
    stop 1, quiet=.true.
  end subroutine perr

  subroutine stop_if_kriging_failed(stage)
    character(*), intent(in) :: stage
    character(len=:), allocatable :: msg

    if (.not. kriging_failed()) return

    ! t_kriging no longer stops internally.  The CLI checks the shared error
    ! state after each kriging call and converts it to a normal program exit.
    msg = kriging_last_error()
    if (len_trim(msg) == 0) msg = 'unknown kriging error'
    call perr('  Error while '//trim(stage)//': '//trim(msg))
  end subroutine stop_if_kriging_failed

  ! ===========================================================================
  ! Diagnostic output (showoptions, showhelp)
  ! ===========================================================================

  subroutine showoptions()
    print "(A)", krig%to_str()
  end subroutine showoptions

  subroutine print_banner()
    print "(A)", ''
    print "(A)", 'SPARKS - Sequential Pilot-point Assisted Random-path Kriging and Simulation'
    print "(13x,A)", 'Version:  '//sparks_version//'  ('//trim(sparks_compiler)//' '//trim(sparks_fc_ver)//')  git: '//trim(sparks_githash)
  end subroutine print_banner

  subroutine showhelp()
    integer, parameter :: Mandatory = 3
    call print_banner()
    print "(A)", ''
    print "(A)", 'Usage:'
    print "(A)", ' sparks -nl sparks.nml  |  sparks -d ndim nobs1 nblock nobs2 ndrift -of obsfile1 [options] [output]'
    print "(A)", '   Perform Kriging or Sequential Gaussian Simulation.'
    print "(A)", '   Developed by mou@sspa.com.'
    print "(A)", ' '
    print "(A)", '   Arguments:'
    do ii = 1, Mandatory
      call print_opt(ii)
    end do
    print "(A)", ' '
    print "(A)", '   Optional arguments:'
    do ii = Mandatory + 1, size(opts)
      call print_opt(ii)
    end do
    print "(A)", '      output'//repeat(' ', 22)// &
      'output file name. "~" or omitted = stdout.'
    print "(A)", ' '
    stop 0, quiet=.true.
  end subroutine showhelp

  ! Word-wrap opts(idx)%description at DESCW characters, breaking on spaces where
  ! possible; fall back to a hard break when no space is found in the window.
  ! The first line is printed with the flag prefix; continuation lines use
  ! blank padding (cont) to align below the description column.
  subroutine print_opt(idx)
    integer, intent(in) :: idx
    integer, parameter  :: DESCW = 68    ! max description width in characters
    integer, parameter  :: PREFLEN = 32  ! width of the flag-prefix column
    character(PREFLEN)  :: pref
    character(PREFLEN)  :: cont
    character(2048)     :: desc
    integer             :: pos, nxt, dlen

    cont = repeat(' ', PREFLEN)
    write (pref, '(A,A2,A,A10,A)') '      -', opts(idx)%short, '  or  --', opts(idx)%name, '     '
    desc = adjustl(trim(opts(idx)%description))
    dlen = len_trim(desc)
    pos = 1
    do while (pos <= dlen)
      nxt = pos + DESCW - 1
      if (nxt < dlen) then
        ! Walk back from nxt to the nearest space for a clean word break.
        do while (nxt > pos .and. desc(nxt:nxt) /= ' ')
          nxt = nxt - 1
        end do
        if (nxt == pos) nxt = pos + DESCW - 1   ! no space found; hard break
      else
        nxt = dlen
      end if
      if (pos == 1) then
        print "(A)", pref//trim(adjustl(desc(pos:nxt)))
      else
        print "(A)", cont//trim(adjustl(desc(pos:nxt)))
      end if
      pos = nxt + 1
      do while (pos <= dlen .and. desc(pos:pos) == ' ')   ! skip leading spaces on next segment
        pos = pos + 1
      end do
    end do
  end subroutine print_opt

end program sparks
