module io
  implicit none

  interface read_data
    module procedure readdata_flat
    module procedure readdata_flat_2d
    module procedure readdata_1int
    module procedure readdata_1real
    module procedure readdata_1real1int
    module procedure readdata_2real
    module procedure readdata_2real1int
    module procedure readdata_3real
  end interface read_data
contains

! ===========================================================================
  ! Data reading helpers
  ! ===========================================================================

  subroutine readdata_flat(file, arr1, ioerr, noheader)
    character(len=*), intent(in) :: file
    real, intent(out) :: arr1(:)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    read (ifile, *, iostat=ioerr) arr1
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_flat

  subroutine readdata_flat_2d(file, arr1, ioerr, noheader)
    character(len=*), intent(in) :: file
    real, intent(out) :: arr1(:,:)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    read (ifile, *, iostat=ioerr) arr1
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_flat_2d

  subroutine readdata_1int(file, n, int1, ioerr, noheader)
    character(len=*), intent(in) :: file
    integer, intent(in) :: n
    integer, intent(out) :: int1(n)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile

    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    read (ifile, *, iostat=ioerr) int1
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_1int

  subroutine readdata_1real(file, n, n1, arr1, ioerr, noheader)
    character(len=*), intent(in) :: file
    integer, intent(in) :: n, n1
    real, intent(out) :: arr1(n1, n)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    integer :: ii, i
    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    do i = 1, n
      read (ifile, *, iostat=ioerr) ii, arr1(:, i)
      if (ioerr /= 0) exit
    end do
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_1real

  subroutine readdata_1real1int(file, n, n1, arr1, int1, ioerr, noheader)
    character(len=*), intent(in) :: file
    integer, intent(in) :: n, n1
    real, intent(out) :: arr1(n1, n)
    integer, intent(out) :: int1(n)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    integer :: ii, i

    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    do i = 1, n
      read (ifile, *, iostat=ioerr) ii, arr1(:, i), int1(i)
      if (ioerr /= 0) exit
    end do
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_1real1int

  subroutine readdata_2real(file, n, n1, n2, arr1, arr2, ioerr, noheader)
    character(len=*), intent(in) :: file
    integer, intent(in) :: n, n1, n2
    real, intent(out) :: arr1(n1, n), arr2(n2, n)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    integer :: ii, i

    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    do i = 1, n
      read (ifile, *, iostat=ioerr) ii, arr1(:, i), arr2(:, i)
      if (ioerr /= 0) exit
    end do
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_2real

  subroutine readdata_2real1int(file, n, n1, n2, arr1, arr2, int1, ioerr, noheader)
    character(len=*), intent(in) :: file
    integer, intent(in) :: n, n1, n2
    real, intent(out) :: arr1(n1, n), arr2(n2, n)
    integer, intent(out) :: int1(n)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    integer :: ii, i

    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    do i = 1, n
      read (ifile, *, iostat=ioerr) ii, arr1(:, i), arr2(:, i), int1(i)
      if (ioerr /= 0) exit
    end do
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_2real1int

  subroutine readdata_3real(file, n, n1, n2, n3, arr1, arr2, arr3, ioerr, noheader)
    character(len=*), intent(in) :: file
    integer, intent(in) :: n, n1, n2, n3
    real, intent(out) :: arr1(n1, n), arr2(n2, n), arr3(n3, n)
    integer, intent(out) :: ioerr
    logical, intent(in), optional :: noheader
    integer    :: ifile
    integer :: ii, i

    open (newunit=ifile, file=trim(file), status='old'); call skip_comment(ifile, ioerr)
    if (.not. present(noheader)) then
      read (ifile, *, iostat=ioerr)
    else if (.not. noheader) then
      read (ifile, *, iostat=ioerr)
    end if
    do i = 1, n
      read (ifile, *, iostat=ioerr) ii, arr1(:, i), arr2(:, i), arr3(:, i)
      if (ioerr /= 0) exit
    end do
    close (ifile)
    if (ioerr /= 0) print *, 'ioerr=', ioerr, " read file = ", trim(file)
  end subroutine readdata_3real

  ! ===========================================================================
  ! Utility helpers (unchanged from original)
  ! ===========================================================================

  function linecount(afile) result(n)
    integer             :: n
    character(len=*), intent(in) :: afile
    integer    :: ifile, ioerr
    open (newunit=ifile, file=trim(afile), status='old'); call skip_comment(ifile, ioerr)
    n = 0
    do
      read (ifile, *, iostat=ioerr)
      if (ioerr /= 0) exit
      n = n + 1
    end do
    close (ifile)
  end function linecount

  subroutine skip_comment(iunit, iostat_out)
    ! Reads past any leading comment/blank lines (starting with #, !, or *)
    ! and leaves the file positioned at the first non-comment line.
    integer, intent(in)  :: iunit
    integer, intent(out) :: iostat_out
    character(1024) :: line
    character(len=3), parameter :: comment = "#!*"
    integer :: i, n
    logical :: is_comment
    n = len_trim(comment)
    do
      read (iunit, '(a)', iostat=iostat_out) line
      if (iostat_out /= 0) return
      line = adjustl(line)
      is_comment = (len_trim(line) == 0)   ! also skip blank lines
      do i = 1, n
        if (line(1:1) /= comment(i:i)) then
          is_comment = .false.
          exit
        end if
      end do
      if (.not. is_comment) then
        backspace (iunit)   ! rewind to the data line
        return
      end if
    end do
  end subroutine
end module
