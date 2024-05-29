unset colorbox
set palette rgb 33,13,10
set size square

size = 60
iterations = 2000

set title "Solution for " . size . "x" . size . " grid with " . iterations . " iterations"
#plot 'solution.dat' binary format='%double' using ($1):($2):($3) with image notitle

plot 'solution.csv' with image notitle