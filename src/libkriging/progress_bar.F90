module progress_bar
contains
subroutine progress(done, total)
  implicit none
  integer, intent(in) :: done, total
  integer, parameter :: width = 30
  integer :: percent, nfill
  character(len=7 + width) :: bar

  if (total <= 0) then
    percent = 100
  else
    percent = max(0, min(100, int(100.0 * real(done) / real(total))))
  end if

  bar = "   % |"
  write(unit=bar(1:3), fmt="(i3)") percent
  nfill = min(width, percent * width / 100)
  if (nfill > 0) bar(7:6+nfill) = repeat("*", nfill)
  bar(7+width:7+width) = "|"

#ifdef __INTEL_COMPILER
  write(unit=6, fmt="(a1,a1,x,a)") '+', char(13), bar
#else
  write(unit=6, fmt="(a1,x,a)", advance="no") char(13), bar
#endif
  flush(unit=6)
end subroutine progress
end module progress_bar
