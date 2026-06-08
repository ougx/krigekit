module progress_bar
contains
subroutine progress(i, n)
  implicit none
  integer        ::i, n
  integer        ::j,k
  character(len=17)::bar="???% |          |"
  j = int(real(i)/real(n) * 100)
  write(unit=bar(1:3),fmt="(i3)") j
  do k=1, int(j/10)
    bar(6+k:6+k)="*"
  enddo
  ! print the progress bar.
#ifdef __INTEL_COMPILER
  write(unit=6,fmt="(a1,a1,x,a17)") '+',char(13), bar
#else
  write(unit=6,fmt="(a1,x,a17)",advance="no") char(13), bar
#endif
  return
end subroutine progress
end module progress_bar
