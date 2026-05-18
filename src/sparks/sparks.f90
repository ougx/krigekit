! Created by Michael Ou
! TODO: drifts
program ppsgs
  use ieee_arithmetic
  use iso_fortran_env, only: input_unit, error_unit, output_unit
  use f90getopt
  !use m_mrgref
  !use m_inssor
  use common
  use rotation
  !use normal_dist
  use variogram
  use kdtree2_module
  use gaussian_quadrature
  use kriging
  implicit none
  character(8), parameter           :: version='20260515'
  type(t_kriging)                   :: krig
  type(option_s), allocatable       :: opts(:)
  character(2048) :: obsfile, obsfile2, gridfile, facfile, blockfile, outfile, randpath, samfile, fomt, rsfile
  character(len=2)  :: opt
  ! inputs
  integer         :: ndim, nobs, ngrid, ndrift, nmax, seed, unbias, nobs2, nmax2, nsim, nblock, iblockpntweight
  real            :: maxdist, vmin, vmax, std, avg, avg2, blocksize(3)
  integer, allocatable :: irandpath(:), iobs(:), igrid(:), iobs2(:), iblock(:,:), inversepath(:)
  logical, allocatable :: kdmask(:)
  real, allocatable ::  obs(:,:),  obs2(:,:), grid(:,:), blocks(:,:), obsdrift(:,:), griddrift(:,:), blockdrift(:,:), ob2drift(:,:)! coordinates and values, variance
  real, allocatable :: samples(:), rangescale(:)

  real, allocatable       ::  solverAP(:), solverB(:)
  integer, allocatable    ::  IPIV(:)

  type(vgm_struct)        :: varios(2,2)
  ! local
  integer         :: matsize, npp, npp1, npp1o, npp1g, npp2, nppd, nppdu, mblock
  integer         :: ifile, ioerr, ifilefac, i, ii, jj, kk, ig, iv, isim, iout, ib, ntp
  integer         :: ivar, jvar, kvar, nvar
  real            :: cov0(2,2), cov00, cov00block, totblockweight
  real, allocatable :: tmpdist(:), tmpdist2(:), xcell(:), tmpdrift(:,:)
  real, allocatable :: weights(:), matA(:,:), rhsB(:)
  integer, allocatable :: inear(:), inear2(:), inearg(:)
  logical         :: correct_weight, writexy, neglect_error, writemat, showargs, loocv, anisotropic_search


  allocate(opts, source=(/&
    option_s("dim"    ,        "d", 5, "define spatial dimension; number of observation; number of grid; number of covariate observation. number of drifts."), &
    option_s("obsfile",       "of", 1, "observation data file. the columns should be id,x(,y,z),obsvalue in that file. drift value columns can be added after obsvalue."), &
    option_s("facfile",       "ff", 1, "interpolation factor file. This file can be created when both 'facfile' and 'gridfile' are defined."), &
    option_s("gridfile",      "gf", 1, "grid file. the columns should be id,x(,y,z) in that file. drift value columns can be added if drift is used."), &
    option_s("blockfile",     "bf", 1, "optional block file for block kriging; the content of this file is the counts of points used for each block, and x(,y,z)."), &
    option_s("blockweight",   "bw", 0, "read point weights for block kriging from the grid file; default is equal weights for each point."), &
    option_s("pathfile",      "pf", 1, "a file contain the indices of the calculation path, if not defined for sinluation, a random path will be generated."), &
    option_s("samfile",       "sf", 1, "a file contain samples from standard normal distribution."), &
    option_s("obsfile2",      "o2", 1, "secondary observation data file for the covariate. the columns should be x,(y,z),obsvalue in that file."), &
    option_s("seed",          "sd", 1, "seed number to generate random path and variance."), &
    option_s("sim",            "s", 0, "enable simulation. default is disable."), &
    option_s("nmax",          "n1", 1, "maximum number of primary variable used for Kriging. default is 200."), &
    option_s("nmax2",         "n2", 1, "maximum number of co-variate used for Kriging. default is 200."), &
    option_s("rangescale",    "rs", 1, "file name that contain the range scaler for each grid cell."), &
    option_s("unbias",         "u", 0, "whether to include unbias term; default is .false. for Simple Kriging; switch on for Ordinary Kriging."), &
    option_s("ang1",          "a1", 1, "set azimuth angle for principal direction; default is 0."), &
    option_s("ang2",          "a2", 1, "set dip angle for principal direction; default is 0."), &
    option_s("ang3",          "a3", 1, "set third rotation angle; default is 0."), &
    option_s("anis1",         "s1", 1, "set first anisotropy ratio; default is 1."), &
    option_s("anis2",         "s2", 1, "set second anisotropy ratio; default is 1."), &
    option_s("vario1",        "v1", 4, "set variogram, following type, range, sill, nugget; type must be sph, exp, gau, pow, cir, hol or lin."), &
    option_s("vario2",        "v2", 4, "set variogram for the secondary covariate."), &
    option_s("varioc",        "vc", 4, "set cross-variogram between the primary and secondary variables."), &
    option_s("bounds",        "bd", 2, "set the lower and upper bounds, if not set, the simulation is unbounded."), &
    option_s("blocksize",     "bs", 1, "set block size for block kriging; 4 × 4 Gaussian quadrature block discretization will be used to assign points and weights."), &
    option_s("maxdist",       "md", 1, "set the maximum distance for search, this is the rotated/anisotropic distance if there is rotation/anisotropy."), &
    option_s("correct",       "cw", 0, "apply weight correction by removing negative weights. default is no weight correction."), &
    option_s("anisosearch",   "as", 0, "neiborgh search is performed based on the projected coordinates using anisotrpy settings. default is non-projected coordinates."), &
    option_s("fmt"    ,       "fm", 1, "fortran format to write results such as '(10F10.3)'; default is '(G0.12)'."), &
    option_s("writexy",       "xy", 0, "write coordinates in the output; default only estimates are written."), &
    option_s("coerce",        "ec", 0, "failed grid point will be set as NaN when it fails to solve equation."), &
    option_s("verbose",        "v", 0, "print running logs to screen."), &
    option_s("loocv"   ,      "cv", 0, "perform Leave-one-out cross-validation (LOOCV)."), &
    option_s("writemat",      "wm", 0, "write the matrix for debugging."), &
    option_s("showargs",      "sa", 0, "show arguments."), &
    option_s("help"   ,        "h", 0, "show this message.") &
  /))


  ! set up the initial/default values
  fomt = '(10(G0.12,:x))'
  outfile = '~'
  obsfile = ''
  obsfile2 = ''
  gridfile = ''
  facfile = ''
  randpath = ''
  samfile = ''
  blockfile = ''
  rsfile = ''
  maxdist = -one
  nmax = 200
  nmax2 = 0
  ndrift = -1
  unbias = 0
  seed = 0
  ang1   = zero
  ang2   = zero
  ang3   = zero
  anis1  = one
  anis2  = one
  nsim = 0
  correct_weight = .false.
  ndim = 2
  nobs = 0
  nobs2 = 0
  ngrid = 0
  nvar = 1
  vmin = -verylarge
  vmax =  verylarge
  writexy = .false.
  verbose = .false.
  neglect_error = .false.
  writemat = .false.
  showargs = .false.
  loocv = .false.
  anisotropic_search=.false.
  iblockpntweight = 0
  blocksize = zero

  ! check if verbose is set
  do
    opt = getopt(opts)
    if (trim(trim(opt)) == 'v') then
      verbose=.true.
      exit
    end if
    if (trim(opt) == char(0)) exit
  end do
  call reset_opt()
  do
    opt = getopt(opts)
    ! print*, opt, trim(optarg)
    select case(trim(opt))
      ! When all options are processed
      case(char(0))
        if (len_trim(optarg)>0) outfile = adjustl(trim(optarg))
        exit

      case( "d")
        read(optarg, *) ndim, nobs, ngrid, nobs2, ndrift
        do i=1, size(opts)
          if (opts(i)%short=="bs") opts(i)%narg = ndim
        end do

      case("of"); obsfile =adjustl(optarg)

      case("o2"); obsfile2=adjustl(optarg)

      case("gf"); gridfile=adjustl(optarg)

      case("bf"); blockfile=adjustl(optarg)

      case("rs"); rsfile=adjustl(optarg)

      case( "u"); unbias = 1

      case("bw"); iblockpntweight = 1

      case("cw"); correct_weight = .true.

      case("ff"); facfile=adjustl(optarg)

      case("pf"); randpath=adjustl(optarg)

      case("sf"); samfile=adjustl(optarg)

      case("sd"); read(optarg, *) seed

      case( "s"); nsim = 1

      case("n1"); read(optarg, *) nmax
      case("n2"); read(optarg, *) nmax2

      case("a1"); read(optarg, *) ang1
      case("a2"); read(optarg, *) ang2
      case("a3"); read(optarg, *) ang3
      case("s1"); read(optarg, *) anis1
      case("s2"); read(optarg, *) anis2
      case("bd"); read(optarg, *) vmin, vmax
      case("bs"); read(optarg, *) blocksize(:ndim)
      case("md"); read(optarg, *) maxdist

      case("v1"); call varios(1,1)%define(optarg)
      case("vc"); call varios(1,2)%define(optarg); call varios(2,1)%define(optarg)
      case("v2"); call varios(2,2)%define(optarg)

      case("fm"); read(optarg, *) fomt

      case("ec"); neglect_error = .true.
      case("xy"); writexy = .true.
      case( "v"); verbose = .true.
      case("cv"); loocv = .true.
      case("as"); anisotropic_search = .true.
      case("wm"); writemat = .true.

      case("sa"); showargs = .true.
      case( "h"); call showhelp

      case default; stop
    end select
    if (ndrift==-1) exit
  end do
  if (ndrift==-1) call perr("  Error: Dimension needs to be defined as the first argument")
  if (showargs) call showoptions()

  ! mem allocations
  if (blocksize(1)>zero) then
    if (blockfile /= "") call perr("  Error: defining both 'blockfile(bf)' or 'blocksize(bs)' is not allowed.")
  end if


  if (loocv) then
    if (nsim>0) call perr("  Error: Simulation should not be activated in cross validation mode.")
    ngrid = nobs
  end if

  allocate(irandpath(ngrid))
  allocate(obs(ndim+1, nobs+ngrid))     ! coordinates plus values
  allocate(iobs(nobs))
  allocate(grid(ndim+1, ngrid))   ! coordinates plus weights for block kriging
  allocate(igrid(ngrid))
  allocate(samples(ngrid))
  if (nobs2>0) then
    nvar = 2
    allocate(obs2(ndim+1, nobs2)) ! coordinates plus values
    allocate(iobs2(nobs2))
  end if
  if (ndrift>0) then
    allocate(obsdrift (ndrift, nobs))
    if (blockfile=="") allocate(griddrift(ndrift, ngrid))
    if (nobs2>0) then
      allocate(ob2drift (ndrift, nobs2))
      ob2drift = zero
    end if
  end if

  call readobs(1)
  if (nobs2>0) call readobs(2)
  if (loocv) then
    ! grid is identical as the observations
    grid(1:ndim,:nobs) = obs(1:ndim,:nobs)
    grid(ndim+1,:) = one
  else if (gridfile /= "") then
    call readgrid()
  end if
  call set_block()
  call random_seed_initialize(seed)
  call set_samples()
  call set_rangescale()

  if (gridfile == '' .and. (.not. loocv)) then
    call calc_by_factor_file
  else
    call kriging_prepare
    call kriging_solve
    if (ifilefac==0) then
      call write_output()
    else
      close(ifilefac)
    end if
  end if
  if (verbose) print*, "PPSGS exited peacefully."

  contains

  subroutine calc_by_factor_file
    ! read the factors from file
    if (verbose) print*, 'Estimating values using factors in "'//trim(facfile)//'"'
    writexy = .false.
    allocate(inear (nobs+nblock-1))
    allocate(weights(nobs+nblock+nobs2-1))
    if (nobs2>0) then
      allocate(inear2(nobs2-1))
    end if
    open(newunit=ifilefac, file=trim(facfile), status='old')
    read(ifilefac, *) ! skip first line
    do ib = 1, nblock
      read(ifilefac, *) irandpath(ib), npp1o, npp1g, npp2, std,           &
        (inear(ii)     , weights(ii),            ii=      1,npp1o+npp1g), &
        (inear2(ii)    , weights(npp1o+npp1g+ii),ii=      1,npp2)
      npp1 = npp1o + npp1g
      npp  = npp1 + npp2
      kk = irandpath(ib)
      call weighted_average(samples(ib), std, blocks(ndim+1:, kk))
    end do
    !if (nsim==0) then
    call open_output()
    call write_output()
    !else
    !  ! put back in correct order
    !  call mrgref(irandpath, inear(1:ngrid))
    !  call write_output(grid(ndim+1, inear(1:ngrid)))
    !end if
  end subroutine calc_by_factor_file

  subroutine kriging_prepare
    integer :: mmax1
    mmax1 = nobs+(ngrid-1)*nsim
    if (loocv) mmax1 = mmax1 - 1
    nsim = min(nsim, 1)
    nmax = min(nmax, mmax1)
    if (nobs2>0) then
      if (nmax2==0) then
        nmax2 = nmax
      end if
      nmax2=min(nmax2, nobs2)
    end if
    call setrot()

    ! Krige the factors
    matsize = nmax+unbias+ndrift+nmax2
    ! if (verbose) print*, "maximum matrix size is", matsize
    allocate(weights(matsize))
    allocate(matA(matsize, matsize), rhsB(matsize), kdnearest(nmax), inear(nmax), tmpdist(nmax), xcell(ndim))
    if (nobs2>0) allocate(kdnearest2(nmax2), inear2(nmax2), tmpdist2(nmax2))

    iblockpntweight = ndim + 1
    if (nsim>0) then
      call set_randpath()
      blocks = blocks(:, irandpath)
      iblock = iblock(:, irandpath)
      if (ndrift>0) blockdrift = blockdrift(:, irandpath)
    else
      irandpath=[(ib, ib=1, nblock)]
      ! for block kriging
      if (blocksize(1)>zero) then
        call set_gaussian_quadrature(ndim, blocksize(:ndim))
        if (allocated(grid)) deallocate(grid)
        ngrid = nblock * ngq
        allocate(grid(ndim+1, ngrid))
        do ib=1, nblock
          do ii=1, ndim
            kk = (ib-1)*ngq
            grid(ii, kk+1:kk+ngq) = gqdelxyz(ii, :) + blocks(ii, ib)
          end do
          grid(iblockpntweight, kk+1:kk+ngq) = gqweight
        end do
        iblock(1,:) = ngq
        iblock(2,:) = [(ib*ngq+1, ib=0, nblock-1)]
        iblock(3,:) = iblock(2,:) -1 + ngq

        ! calculate the cov00
        cov00block = zero
        gqdelxyz(1:ndim, :) = rotate(ndim, ngq, gqdelxyz(1:ndim, :))
        do ii=1, ngq
          cov00block = cov00block + gqweight(ii) * sum(varios(1,1)%covfuc(sdistn1(gqdelxyz(1:ndim, 1:ngq), gqdelxyz(1:ndim, ii))) * gqweight(1:ngq))
        end do
      end if
    end if

    ! store the inverse path to put back the grid in original order
    allocate(inversepath(nblock))
    do ii=1, nblock
      inversepath(irandpath(ii)) = ii
    end do

    ! calculate distance
    call set_dist()
    if (facfile/='') then
      if (verbose) print*, 'Factors written in "'//trim(facfile)//'"'
      open(newunit=ifilefac, file=trim(facfile), status='replace')
      write(ifilefac, '(A,*(:" index",I0," weight",I0))') 'igrid nobs1 ngrid nobs2 std', (ii,ii,ii=1,nmax+nmax2)
    else
      ifilefac = 0
      call open_output()
    end if

    allocate(solverAP(matsize*(matsize+1)/2), solverB(matsize), IPIV(matsize))

  end subroutine kriging_prepare

  subroutine kriging_solve
    ! initialization
    mblock = 0
#ifdef __INTEL_COMPILER
    if (verbose) open (unit=6, carriagecontrol='fortran')
#endif

    do ib = 1, nblock
      ! print*, "krige1100 ", ib, igrid(ig)
      if (verbose) call progress(real(ib)/real(nblock))
      ! print*, "krige1200 ", "search for the nearest obs"
      if (anisotropic_search) then
        xcell = robs(:ndim,nobs+ib)
      else
        xcell = obs(:ndim,nobs+ib)
      end if
      if (loocv) then
        kdmask(ib) = .false.          ! the Leave-One-Out Cross Validation excludes the observation point itself
        if (ib>1) kdmask(ib-1)=.true. ! turn back on the last observation point
      end if
      if (nmax<nobs+mblock) then
        ! print*, "krige1205", "search for the nearest obs"
        call kdtree2_n_nearest(obstree,xcell,nmax,kdnearest,kdmask)
        ! print*, "krige1210", "search for the nearest obs", kdmask(:10)
        npp1o = 0
        npp1g = 0
        npp1 = nmax
        do ii = 1, nmax
          if (kdnearest(ii)%idx<=nobs) then
            npp1o = npp1o + 1
            inear(npp1o) = kdnearest(ii)%idx
          else
            npp1g = npp1g + 1
            inear(nmax-npp1g+1) = kdnearest(ii)%idx
          end if
        end do
      else
        ! use all points
        npp1o = nobs
        npp1g = mblock
        npp1 = nobs + mblock
        if (nsim>0) then
          inear(1:npp1) = pack([(ii,ii=1,nobs+ib-1)], kdmask(1:nobs+ib-1))
        else
          if (ib ==1) inear(1:npp1) = [(ii, ii=1,nobs)]
        end if
      end if

      ! print*, "krige1300", "search for the nearest obs", kdmask(:10)
      if (nobs2>0) then
        if (nmax2<nobs2) then
          ! print*, "krige18001 ", 'search for neighbor2'
          npp2 = nmax2
          call kdtree2_n_nearest(obstree2,xcell,nmax2,kdnearest2)
          inear2(1:npp2) = kdnearest2%idx
        else
          npp2 = nobs2
          if (ib==1) inear2(1:npp2) = [(ii,ii=1,npp2)]
        end if
      else
        npp2 = 0
      end if
      ! print*, "krige1400 calculate the distance"
      if (anisotropic_search) then
        tmpdist(1:npp1)               = sdistn1(robs(:ndim,inear (1:npp1)), xcell)
        if (nobs2>0) tmpdist2(1:npp2) = sdistn1(rob2(:ndim,inear2(1:npp2)), xcell)
      else
        tmpdist(1:npp1)               = sdistn1(obs (:ndim,inear (1:npp1)), xcell)
        if (nobs2>0) tmpdist2(1:npp2) = sdistn1(obs2(:ndim,inear2(1:npp2)), xcell)
      end if

      ! print*, "krige1500 validate data for colocation and maxdist"
      if (any(tmpdist(1:npp1)<verysmall)) then
        ! if there is an observation point very close to the unknown location. use that point
        inear(1:1) = inear(minloc(tmpdist(1:npp1)))
        weights=zero
        weights(1) = one
        npp1  = 1
        npp1o = 1
        npp1g = 0
        npp2  = 0
        nppd = 1
        nppdu = 1
        npp = 1
        std = sqrt(varios(1,1)%nugget) ! measurement error
      else
        ! calculate the distance from the unknown grid cell to all data points
        if (maxdist>0) then
          npp   = count(tmpdist(1:npp1)<=maxdist)
          if (npp /= npp1) then
            inear(1:npp)   = pack(inear  , tmpdist<=maxdist)
            tmpdist(1:npp) = pack(tmpdist, tmpdist<=maxdist)
            npp1  = npp
            npp1o = count(inear<=nobs)
            npp1g = npp1 - npp1o
          end if
          if (nobs2>0) then
            npp   = count(tmpdist2(1:npp2)<=maxdist)
            if (npp /= npp2) then
              npp2 = npp
              inear2(1:npp2)   = pack(inear2  , tmpdist2<=maxdist)
              tmpdist2(1:npp2) = pack(tmpdist2, tmpdist2<=maxdist)
            end if
          end if
        end if

        ! print*, "krige1600 calculate the data input counts"
        npp = npp1 + npp2       ! total obs points
        nppd  = npp + ndrift    ! obs points + drifts
        nppdu = nppd + unbias   ! obs points + drifts + unbias

        ! construct linear system
        matA = zero
        rhsB = zero
        ! print "(99I9)", nobs, nobs2, ngrid, nmax2, npp2, npp1o, npp2, ndrift, unbias
        ! print*, 'start to build matrix', sdistn1(robs(:,inear (1+1:npp1o)  ), robs(:,1))
        ! obs1
        totblockweight = sum(grid(iblockpntweight, iblock(2,ib):iblock(3,ib)))
        grid(iblockpntweight, iblock(2,ib):iblock(3,ib)) = grid(iblockpntweight, iblock(2,ib):iblock(3,ib)) / totblockweight
        do ii=1, npp1
          xcell = robs(:,inear(ii))
          matA(ii         , ii) = cov0(1,1)                                                                           ! print*, ii, "diagonal"
          matA(ii  +1:npp1, ii) = varios(1,1)%covfuc(sdistn1(robs (:,inear (ii+1:npp1)), xcell)/rangescale(ib))          ! print*, ii, "obs1 ~ obs1&grid"
          rhsB(ii             ) = sum( &
           varios(1,1)%covfuc(sdistn1(rgrid(:,iblock(2,ib):iblock(3,ib)), xcell)/rangescale(ib)) * &
           grid(iblockpntweight, iblock(2,ib):iblock(3,ib)) &
          )
          ! print*, ii, "right hand side/block mean", sdistn1(rgrid(:,iblock(2,ib):iblock(3,ib)), xcell)
          ! print*, ii, "right hand side/block mean", rangescale(ib)
          ! print*, ii, "right hand side/block mean", sdistn1(rgrid(:,iblock(2,ib):iblock(3,ib)), xcell)/rangescale(ib)
          ! print*, ii, "right hand side/block mean", varios(1,1)%covfuc(sdistn1(rgrid(:,iblock(2,ib):iblock(3,ib)), xcell)/rangescale(ib))
          ! print*, ii, "right hand side/block mean", grid(iblockpntweight, iblock(2,ib):iblock(3,ib))
          ! print*, ii, "right hand side/block mean", rhsB(ii)
          ! stop
        end do

        ! obs2
        do kk=1, npp2
          ii = kk + npp1
          xcell = rob2(:,inear2(kk))
          matA(ii, 1:npp1) = varios(1,2)%covfuc(sdistn1(robs(:,inear(1:npp1)), xcell))                                ! print*, ii, "obs1 ~ obs2"

          matA(ii      ,ii) = cov0(2,2)                                                                            ! print*, ii, "diagonal"
          matA(ii+1:npp,ii) = varios(2,2)%covfuc(sdistn1(rob2(:,inear2(kk+1:npp2)), xcell))                          ! print*, ii, "obs2 ~ obs2"
          rhsB(ii)          = sum( &
           varios(1,2)%covfuc(sdistn1(rgrid(:,iblock(2,ib):iblock(3,ib)), xcell)/rangescale(ib)) * &
           grid(iblockpntweight, iblock(2,ib):iblock(3,ib)) &
          )
          ! if (kk==1) then
          !   ii = kk + npp1
          !   print*
          !   print*, ii, "distance  ", sdistn1(rob2(:,inear2(kk+1:kk+1)), xcell)
          !   print*, ii, "rangescale", rangescale(ib)
          !   print*, ii, "covariance", varios(2,2)%covfuc(sdistn1(rob2(:,inear2(kk+1:kk+1)), xcell))
          !   print*, varios(2,2)%vgms(1)%vgm%tostr()
          !   stop
          ! end if
        end do

        ! drift
        if (ndrift > 0) then
          matA(npp+1:nppd,       :npp1o) = obsdrift(:, inear(1:npp1o))                                              ! print*, "obs drift"
          matA(npp+1:nppd,npp1o+1:npp1 ) = blockdrift(:, inear(npp1o+1:npp1)-nobs)                                  ! print*, "grid drift"
          matA(npp+1:nppd,npp1 +1:npp  ) = ob2drift(:, inear2(1:npp2))                                              ! print*, "obs2 drift"
          rhsB(npp+1:nppd  ) = blockdrift(:,ib)
        end if

        ! unbias
        if (unbias > 0) then
          matA(nppdu, :npp) = one
          rhsB(nppdu      ) = one
        end if

        ! calculate the weights
        call solve_matrix()

        ! print*, "weights", weights(:npp)
        if (correct_weight) then
          where(weights(1:npp)<zero) weights(1:npp)=zero
          weights(1:npp) = weights(1:npp) / sum(weights(1:npp))
        end if

        if (iblock(1, ib)<2) then
          cov00 = cov0(1,1)
        else if(blocksize(1)>zero) then
          cov00 = cov00block
        else
          cov00 = zero
          do ii=iblock(2,ib), iblock(3,ib)
            cov00 = cov00 + grid(iblockpntweight, ii) * &
                    sum(varios(1,1)%covfuc(sdistn1(rgrid(:,iblock(2,ib):iblock(3,ib)), rgrid(1:ndim,ii))/rangescale(ib)) * grid(iblockpntweight, iblock(2,ib):iblock(3,ib)))
          end do
        end if
        std = sqrt(max(cov00 - sum(rhsB(1:nppdu) * weights(1:nppdu)), zero))
      end if
      !============== save results ==============
      inear(npp1o+1:npp1) = inear(npp1o+1:npp1) - nobs
      if (ifilefac==0) then
        ! save result to grid for exporting
        call weighted_average(samples(ib), std, blocks(ndim+1:, ib))
        ! save results to csv
        if (writexy) write(iout, "(I0,*(:',',G0.12))") irandpath(ib), blocks(:, ib)
      else
        ! save results to factor file
        write(ifilefac, '(4(I0,x),G0.12,*(:x,I0,x,F0.10))') irandpath(ib), npp1o, npp1g, npp2, std, &
          (inear(ii)           , weights(ii),     ii=      1,npp1o), &
          (irandpath(inear(ii)), weights(ii),     ii=npp1o+1,npp1),  &
          (inear2(ii)          , weights(npp1+ii),ii=      1,npp2)
      end if
      ! if (ig==3) stop
      if ((nsim>0) .and. .not. isnan(blocks(ndim+1, ib))) then
        kdmask(nobs+ib) = .true.
        mblock = mblock + 1
      end if
    end do
#ifdef __INTEL_COMPILER
    if (verbose) close(6)
#else
    if (verbose) print *, "" ! start a new line below the progress bar
#endif
  end subroutine kriging_solve

  function linecount(afile)
    integer         :: linecount
    character(1024) :: afile
    open(newunit=ifile, file=trim(afile), status='old')
    linecount = 0
    do
      read(ifile,*, iostat=ioerr)
      if (ioerr/=0) EXIT
      linecount = linecount + 1
    end do
    close(ifile)
  end function

  !@(#) M_random::scramble(3f): return integer array of random values 1 to N.
  ! https://fortranwiki.org/fortran/show/scramble
  function scramble( number_of_values ) result(array)

    integer,intent(in)    :: number_of_values
    integer,allocatable   :: array(:)
    integer               :: l, j, k, m, n
    integer               :: temp
    real                  :: u

    array=[(l,l=1,number_of_values)]

    ! The intrinsic RANDOM_NUMBER(3f) returns a real number (or an array
    ! of such) from the uniform distribution over the interval [0,1). (ie.
    ! it includes 0 but not 1.).
    !
    ! To have a discrete uniform distribution on
    ! the integers {n, n+1, ..., m-1, m} carve the continuous distribution
    ! up into m+1-n equal sized chunks, mapping each chunk to an integer.
    !
    ! One way is:
    !   call random_number(u)
    !   j = n + FLOOR((m+1-n)*u)  ! choose one from m-n+1 integers

    n=1
    m=number_of_values
    do k=1,2
        do l=1,m
          call random_number(u)
          j = n + FLOOR((m+1-n)*u)
          ! switch values
          temp=array(j)
          array(j)=array(l)
          array(l)=temp
        enddo
    enddo
  end function scramble

  ! search for the k smallest values in an unsorted array, return the index of these numbers
  function search_smallest(vals, n, nk) result(idx_nearest)
    integer         :: n, nk
    real            :: vals(n)
    integer         :: idx_nearest(nk)
    ! local
    real            :: valmax
    integer         :: imax, k

    idx_nearest = [(k,k=1, nk)]

    imax = maxloc(vals(:nk), dim=1)
    valmax = vals(imax)

    do k = nk+1, n
      if(vals(k)<valmax) then
        ! print "(I10,12F10.1)", k,vals(k),valmax,vals(idx_nearest)
        idx_nearest(imax) = k
        imax = maxloc(vals(idx_nearest), dim=1)
        valmax = vals(idx_nearest(imax))
      end if
    end do
  end function

  subroutine readdata(afile, nn, idx, arr, arr1)
    character(1024)             :: afile
    integer                     :: nn
    integer                     :: idx(:)
    real            :: arr(:,:)
    real, optional  :: arr1(:,:)
    open(newunit=ifile, file=trim(afile), status='old')
    read(ifile,*, iostat=ioerr)
    do ii = 1, nn
      if (present(arr1)) then
        read(ifile,*, iostat=ioerr) idx(ii), arr(:, ii), arr1(:, ii)
      else
        read(ifile,*, iostat=ioerr) idx(ii), arr(:, ii)
      end if
    end do
    close(ifile)
  end subroutine readdata

  function readdata_1col(afile, nn) result(arr)
    character(1024) :: afile
    integer         :: nn
    real:: arr(nn)
    open(newunit=ifile, file=trim(afile), status='old')
    read(ifile,*, iostat=ioerr)
    read(ifile,*, iostat=ioerr) arr
    close(ifile)
  end function readdata_1col

  subroutine readgrid()
    ! reading grid file
    if (verbose) print*, 'Reading GRID in "'//trim(gridfile)//'"'
    if (ngrid==0) ngrid = linecount(gridfile) - 1
    if (allocated(griddrift)) then
      call readdata(gridfile, ngrid, igrid, grid(1:ndim+iblockpntweight,:), griddrift)
    else
      call readdata(gridfile, ngrid, igrid, grid(1:ndim+iblockpntweight,:))
    end if
    if (iblockpntweight==0) grid(ndim+1,:) = one
  end subroutine readgrid

  subroutine set_block()
    ! reading blocks file
    if (blockfile=="") then
      nblock = ngrid
      allocate(blocks(ndim+2+nsim, ngrid))
      allocate(iblock(3, nblock))
      blocks(:ndim, :) = grid(:ndim, :)
      if (ndrift>0) allocate(blockdrift, source=griddrift)
      iblock(1,:) = 1
      iblock(2,:) = [(ig, ig=1, ngrid)]
      iblock(3,:) = iblock(2,:)
    else
      if (verbose) print*, 'Reading blocks in "'//trim(blockfile)//'"'
      nblock = linecount(blockfile) - 1
      nblock=min(nblock, ngrid)
      allocate(iblock(3, nblock), blocks(ndim+2, nblock))
      if (ndrift>0) then
        allocate( blockdrift(ndrift, nblock) )
        call readdata(blockfile, nblock, iblock(1,:), blocks(1:ndim,:), blockdrift)
      else
        call readdata(blockfile, nblock, iblock(1,:), blocks(1:ndim,:))
      end if
    end if
    iblock(2,1) = 1
    iblock(3,1) = iblock(1,1)
    do ib=2, nblock
      iblock(2,ib) = iblock(3,ib-1)+1
      iblock(3,ib) = iblock(2,ib  )+iblock(1,ib)-1
    end do
  end subroutine set_block

  subroutine readobs(ivar)
    ! reading OBS file
    integer     :: ivar
    if (ivar==1) then
      if (verbose) print*, 'Reading OBS in "'//trim(obsfile)//'"'
      if (nobs==0) nobs = linecount(obsfile) - 1
      if (ndrift>0) then
        call readdata(obsfile, nobs, iobs, obs, obsdrift)
      else
        call readdata(obsfile, nobs, iobs, obs)
      end if
      !do ii =1, 5; print*,obs(:,ii); end do
    else
      if (verbose) print*, 'Reading covariate OBS in "'//trim(obsfile2)//'"'
      if (nobs2==0) nobs2 = linecount(obsfile2) - 1
      if (ndrift>0) then
        call readdata(obsfile2, nobs2, iobs2, obs2, ob2drift)
      else
        call readdata(obsfile2, nobs2, iobs2, obs2)
      end if
      !do ii =1, 5; print*,obs2(:,ii); end do
    end if
  end subroutine readobs

  subroutine set_randpath()
    if(nsim==0) then
      ! pass
    else
      if (randpath == '') then
        if (verbose) print*, 'Generating random path'
        irandpath = scramble(nblock)
        open(newunit=ifile, file=trim(facfile)//'_path', status='replace')
        write(ifile, '(A)') 'randpath'
        do ii=1, nblock; write(ifile, '(I0)') irandpath(ii); end do
        close(ifile)
      else
        if (verbose) print*, 'Reading random path in "'//trim(randpath)//'"'
        irandpath = int(readdata_1col(randpath, nblock))
      end if
    end if
  end subroutine set_randpath

  subroutine set_rangescale()
    allocate(rangescale(nblock))
    if(rsfile=='') then
      rangescale = 1.0
    else
      open(newunit=ifile, file=trim(rsfile), status='old')
      read(ifile, *) rangescale
      close(ifile)
    end if
  end subroutine set_rangescale

  subroutine set_samples()
    if(nsim==0) then
      if (verbose) print*, 'No simulation. All samples are set to 0.'
      samples = zero
    else
      if (samfile == '') then
        if (verbose) print*, 'Generating samples'
        call r8vec_normal_01 ( nblock, samples )
        open(newunit=ifile, file=trim(facfile)//'_sample', status='replace')
        write(ifile, '(A)') 'sample'
        do ii=1, nblock; write(ifile, '(G0.10)') samples(ii); end do
        close(ifile)
      else
        if (verbose) print*, 'Reading samples in "'//trim(samfile)//'"'
        samples = readdata_1col(samfile, nblock)
      end if
    end if
  end subroutine set_samples

  ! calculate distance between multiple points to one point
  function sdist1(coord1, coord2) result(res)
    real              :: coord1(:), coord2(:)
    real              :: res
    res = sqrt(sum((coord1(1:ndim) - coord2(1:ndim)) ** 2))
  end function

  function sdistn1(coords, coord2) result(res)
    real              :: coords(:,:), coord2(:)
    real, allocatable :: res(:)
    ! local
    integer                       :: n1
    n1 = size(coords, dim=2)
    res = [(sdist1(coords(:, ii), coord2), ii = 1, n1)]
  end function

  ! read factor file
  subroutine readfactor()
    if (verbose) print*, 'Read factors in "'//trim(facfile)//'"'
    open(newunit=ifile, file=trim(facfile), status='old')
    close(ifile)
  end subroutine readfactor

  subroutine perr(msg)
    character(*) :: msg
    write(error_unit, '(A)') msg
    stop
  end subroutine perr

  subroutine set_dist()
    real :: centerloc(ndim)
    if (verbose) print*, 'Building distance kdtree ...'

    ! rotating/scaling
    centerloc = sum(blocks(1:ndim, 1:nblock), dim=2)/nblock
    obs(:ndim,nobs+1:) = blocks(1:ndim, :) ! the order of blocks has been shuffled
    robs = rotate(ndim, nobs+nblock, obs(:ndim,:), centerloc)
    if (blockfile=="" .and. blocksize(1)==zero) then
      rgrid = robs(:,inversepath+nobs)
    else
      rgrid = rotate(ndim, ngrid, grid(1:ndim, :), centerloc)
    end if
    allocate(kdmask(nobs+nblock*nsim))
    kdmask(:nobs)   = .true.
    kdmask(nobs+1:) = .false.
    if (anisotropic_search) then
      obstree => kdtree2_create(robs(:ndim,:nobs+nsim*nblock), sort=.false., rearrange=.false.)
    else
      obstree => kdtree2_create(obs (:ndim,:nobs+nsim*nblock), sort=.false., rearrange=.false.)
    end if

    if (nobs2>0) then
      rob2 = rotate(ndim, nobs2, obs2(1:ndim, 1:nobs2), centerloc)
      if (anisotropic_search) then
        obstree2 => kdtree2_create(rob2(1:ndim,1:nobs2), sort=.false., rearrange=.false.)
      else
        obstree2 => kdtree2_create(obs2(1:ndim,1:nobs2), sort=.false., rearrange=.false.)
      end if
    end if
  end subroutine set_dist


  subroutine set_matrix()
    ! allocate matrix storage
    integer :: nppindex(0:2), nppsize(1:2), inears(npp1+npp2)
    integer :: ii0
    nppindex(0) = 0
    nppindex(1) = npp1
    nppindex(2) = npp1+npp2
    nppsize(1)  = npp1
    nppsize(2)  = npp2
    ! print*, "krige1600 calculate the data input counts"
    npp = npp1 + npp2       ! total obs points
    nppd  = npp + ndrift    ! obs points + drifts
    matsize = nppd + unbias   ! obs points + drifts + unbias

    ! construct linear system
    matA = zero
    rhsB = zero

    do ivar=1, nvar
      ! auto covariance for ivar
      do ii=1, nppsize(ivar)
        kk = nppindex(ivar-1) + ii
        matA(kk                 , kk) = cov0(ivar, ivar)
        matA(kk+1:nppindex(ivar), kk) = varios(ivar,ivar)%covfuc(sdistn1(robs(:,inear(ii+1:nppsize(ivar))), robs(:,inear(ii)))/rangescale(ib))
      end do
      do jvar=ivar+1, nvar
        ! cross covariance between ivar and jvar
      end do

    end do
  end subroutine set_matrix


  subroutine solve_matrix()
    integer                 ::  INFO
    character*256           ::  sig
    ! nppdu is the mat size ~ obs + drift + unbias

    ! store the matrix to column compact form
    kk = 1
    do ii=1, nppdu
      solverAP(kk:kk+ii-1) = matA(ii, 1:ii)
      kk = kk + ii
    end do
    solverB = rhsB(1:nppdu)
    call SSPSV( 'U', nppdu, 1, solverAP, IPIV, solverB, nppdu, INFO )

    if (INFO /= 0) then
      weights = ieee_value(weights(1), ieee_signaling_nan)
      if (.not. neglect_error) then
        call write_matrix()
        write(sig, "(I0)") ib
        call perr(new_line("")//'Failed to find solution of the linear system at cell '//trim(sig))
      end if
    else
      if (correct_weight) then
        weights(1:nppdu) = merge(solverB(1:nppdu), 0.0, solverB(1:nppdu)>0)
        weights(1:nppdu) = weights(1:nppdu) / sum(weights(1:nppdu))
      else
        weights(1:nppdu) = solverB(1:nppdu)
      end if
    end if
    if (writemat) call write_matrix()
  end subroutine solve_matrix

  subroutine open_output()
    character :: cname(3)=['x', 'y', 'z']
    if (trim(outfile)=='~') then
      iout = output_unit ! print to stdout
    else
      open(newunit=iout, file=trim(outfile), status='replace')
    end if
    if (writexy) then
      if (nsim==0) then
        write(iout, '(99(A,:,","))') 'igrid', cname(1:ndim), 'estimate', 'std'
      else
        write(iout, '(99(A,:,","))') 'igrid', cname(1:ndim), 'estimate', 'std', 'kriged'
      end if
    end if
  end subroutine open_output

  subroutine write_output()
    if (.not. writexy) write(iout, fomt) blocks(ndim+1, :)
    if (iout /= output_unit) close(ifile)
    if (verbose) print*, 'Results have been written successfully'
  end subroutine write_output

  subroutine write_matrix()
    character(len=20) :: sig
    character(len=6 ) :: cname(3)=['x_orig', 'y_orig', 'z_orig']
    character(len=9 ) :: rname(3)=['x_rotated', 'y_rotated', 'z_rotated']

    write(sig, "(I0)") irandpath(ib)
    open(newunit=ifile, file='matA_'//trim(sig)//'.dat', status='replace')
    do ii =1, nppdu
      write(ifile, "(*(ES15.7))") matA(:nppdu, ii)
    end do
    close(ifile)
    open(newunit=ifile, file='rhsB_'//trim(sig)//'.dat', status='replace')
    do ii =1, nppdu
      write(ifile, "(*(ES15.7))") rhsB(ii)
    end do
    close(ifile)

    open(newunit=ifile, file='data_'//trim(sig)//'.csv', status='replace')
    write(ifile, '(99(A,:,","))') 'source','id', cname(1:ndim), 'value', rname(1:ndim), 'dist_rotated', 'weight'
    do ii =1, npp1o
      write(ifile, "('obs ,',I0,*(:,',',ES15.8))") inear(ii),obs(:ndim+1,inear(ii)), robs(1:ndim,inear(ii)),tmpdist(ii),weights(ii)
    end do
    do ii=npp1o+1, npp1
      kk = inear(ii)-nobs
      write(ifile, "('grid ,'I0,*(:,',',ES15.8))") irandpath(kk),blocks(:ndim+1, kk), robs(1:ndim,inear(ii)),tmpdist(ii),weights(ii)
    end do
    do ii =1, npp2
      write(ifile, "('obs2,',I0,*(:,',',ES15.8))") inear2(ii),obs2(:ndim+1,inear2(ii)),rob2(1:ndim,inear2(ii)),tmpdist2(ii),weights(npp1+ii)
    end do
    close(ifile)
    ! call print_rotmat
  end subroutine write_matrix

  subroutine weighted_average(sample, std, res)
    real              :: res(:) ! result array, mean and standard deviation
    real              :: sample, std
    ! real, allocatable :: obsval(:)

    res(1) = zero
    res(2) = std

    if (any(isnan(weights(1:npp)))) then
      res = ieee_value(res, ieee_signaling_nan)
    else
      ! obsval = [obs(ndim+1,inear(:npp1o)), blocks(ndim+1, inear(npp1o+1:npp1)), obs2(ndim+1, inear2(1:npp2))]
      res(1) = sum(weights(      1:npp1o) * obs   (ndim+1, inear(       :npp1o))) + \
               sum(weights(npp1o+1:npp1 ) * blocks(ndim+1, inear(npp1o+1:npp1 )))
      if ((npp2>0 .or. unbias==0) .and. npp1>0) then
        avg = sum([obs(ndim+1,inear(:npp1o)), blocks(ndim+1, inear(npp1o+1:npp1))]) / npp1
      else
        avg = zero
      end if
      if (npp2>0) then
        ! ISAAKS and SRIVASTAVA, An Introduction to Applied Geostatistics, pp410
        avg2 = sum(obs2(ndim+1, inear2(1:npp2))) / npp2
        res(1) = res(1) + sum(weights(npp1+1:npp) * (obs2(ndim+1, inear2(1:npp2)) + avg - avg2))
      end if
      ! res(1) = sum(weights(1:npp) * obsval(1:npp))
      ! if (isnan(res(1))) then
      !   print*, ib, irandpath(ib), npp1o, npp1, npp2
      !   print*, "Weights"
      !   print "(10F10.6)", weights(1:npp)
      !   print*, "Values"
      !   print "(10F10.6)", obsval(1:npp)
      !   print*, "inear1"
      !   print "(10I10)", inear (1:npp1)
      !   print*, "inear2"
      !   print "(10I10)", inear2(1:npp2)
      !   print*, "obs1"
      !   print "(10F10.6)", obs(ndim+1,inear(:npp1o))
      !   print*, "grid obs"
      !   print "(10F10.6)", blocks(ndim+1, inear(npp1o+1:npp1))
      !   print*, "obs2"
      !   print "(10F10.6)", obs2(ndim+1, inear2(1:npp2))
      !   stop
      ! end if
      if (nsim>0) res(3) = res(1)
      ! print*, "res0   ", res
      ! print*, "weights", weights(1:npp)
      ! print*, "obsval ", obsval(1:npp)

      ! Simple Kriging: Mean Correction
      if (unbias==0) res(1) = res(1) + (one - sum(weights(1:npp))) * avg
      ! print*, "res1   ", res
      ! Simulation
      if (nsim>0) res(1) = res(1) + sample * std
      res(1) = max(min(res(1), vmax), vmin)
      ! print*, "res2   ", res
      ! deallocate(obsval)
    end if
  end subroutine

  function yesno(condition) result(res)
    character*3  :: res
    logical      :: condition
    if (condition) then
      res = "Yes"
    else
      res = "No"
    end if
  end function yesno

  subroutine showoptions()

    print "(A   )", ""
    print "(A   )", "==================== Configuration ===================="
    print "(A,A)",  ' Version                : ', version
    print "(A,A )", " Observation file 1     : ", trim(obsfile)
    print "(A,A )", " Observation file 2     : ", trim(obsfile2)
    print "(A,A )", " grid file              : ", trim(gridfile)
    print "(A,A )", " block file             : ", trim(blockfile)
    print "(A,A )", " path file              : ", trim(randpath)
    print "(A,A )", " sample file            : ", trim(samfile)
    print "(A,A )", " factor file            : ", trim(facfile)
    print "(A,A )", " Output file            : ", trim(outfile)
    print "(A,I0)", " Number of dimensions   : ", ndim
    print "(A,I0)", " Number of observations : ", nobs
    print "(A,I0)", " Number of grid cells   : ", ngrid
    print "(A,I0)", " Number of covariate obs: ", nobs2
    print "(A,I0)", " Number of drifts       : ", ndrift
    print "(A,I0)", " Seed number            : ", seed
    print "(A,I0)", " nmax1                  : ", nmax
    print "(A,I0)", " nmax2                  : ", nmax2
    print "(A,A )", " enable simulation      : ", yesno(nsim==1)
    print "(A,A )", " unbias                 : ", yesno(unbias==1)
    print "(A,A )", " weight correction      : ", yesno(correct_weight)
    print "(A,A )", " Neglect error          : ", yesno(neglect_error)
    print "(A,A )", " LOO-Cross Validation   : ", yesno(loocv)
    print "(A,A )", " Project coordinates    : ", yesno(anisotropic_search)
    print "(A,A )", " Write CSV with coords  : ", yesno(writexy)
    print "(A,A )", " Write matrix for debug : ", yesno(writemat)
    print "(A,A )", " Verbose                : ", yesno(verbose)
    print "(A,G0)", " ang1                   : ", ang1
    print "(A,G0)", " ang2                   : ", ang2
    print "(A,G0)", " ang3                   : ", ang3
    print "(A,G0)", " anis1                  : ", anis1
    print "(A,G0)", " anis2                  : ", anis2
    print "(A,G0)", " vmin                   : ", vmin
    print "(A,G0)", " vmax                   : ", vmax
    print "(A,G0)", " maxdist                : ", maxdist
    do iv=1, varios(1,1)%nstruct; print        *,  "variogram model 1      : ", varios(1,1)%vgms(iv)%vgm%tostr(); end do
    do iv=1, varios(1,2)%nstruct; print        *,  "variogram model cross  : ", varios(1,2)%vgms(iv)%vgm%tostr(); end do
    do iv=1, varios(2,2)%nstruct; print        *,  "variogram model 2      : ", varios(2,2)%vgms(iv)%vgm%tostr(); end do
    print "(A,A )", " Output format          : ", trim(fomt)
    print "(A   )", "================== End Configuration =================="
    print "(A   )", ""
  end subroutine showoptions

  subroutine showhelp()
    integer, parameter    :: Mandatory = 2
    print "(A)", ''
    print "(A)", ''
    print "(A)", ' ppsgs -d ndim nobs ngrid nobs2 ndrift -of obsfile [-gf gridfile] [-ff facfile] [-o2 obsfile2] [options] [output]'
    print "(A)", ''
    print "(A)", '   Perform Kriging or Sequential Gaussian Simulation.'
    print "(A)", '   Developed by mou@sspa.com.'
    print "(A)", '   Version: '//version//'.'
    print "(A)", ' '
    print "(A)", '   Arguments:'
    do ii=1, Mandatory
      print "(A)", '      -'//opts(ii)%short//'  or  --'//opts(ii)%name(:10)//'  '//trim(opts(ii)%description)
    end do
    print "(A)", ' '
    print "(A)", '   Optional arguments:'
    do ii=Mandatory+1, size(opts)
      print "(A)", '      -'//opts(ii)%short//'  or  --'//opts(ii)%name(:10)//'  '//trim(opts(ii)%description)
    end do
    print "(A)", '      output                 output file name. If output=="~" or omitted, output will be printed on screen.'
    print "(A)", ' '
    stop
  end subroutine showhelp

end program
