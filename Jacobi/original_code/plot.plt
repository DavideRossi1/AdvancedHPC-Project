unset colorbox
set palette rgb 33,13,10
set size square
set terminal pngcairo size 640,480
set output 'solution.png'
plot 'solution.dat' with image notitle
