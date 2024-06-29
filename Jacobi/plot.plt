unset colorbox
set palette rgb 33,13,10
set size square

if (exists("NITER")) { niter=NITER } else { niter = 1 }
if(niter>1) {
    set terminal gif animate delay 5 size 640,480
    set output "output/solution.gif"
    do for [i=0:niter-1] {
        set title sprintf("Frame %d of %d", i, niter)
        plot sprintf('output/solution%d.dat', i) binary format='%double' using ($1):($2):($3) with image notitle
    }
} else {
    set terminal pngcairo size 640,480
    set output 'output/solution.png'
    plot 'output/solution0.dat' binary format='%double' using ($1):($2):($3) with image notitle
}
