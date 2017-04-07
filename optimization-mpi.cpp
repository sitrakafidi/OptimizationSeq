/*
  Branch and bound algorithm to find the minimum of continuous binary 
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include <ctime>
#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <mpi.h>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"
#include <omp.h>

using namespace std;


// Split a 3D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
	       interval &xl, interval& xr, interval& yl, interval& yr)
{
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(),xm);
  xr = interval(xm,x.right());
  yl = interval(y.left(),ym);
  yr = interval(ym,y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,  // Function to minimize
        const interval& x, // Current bounds for 1st dimension
        const interval& y, // Current bounds for 2nd dimension
        double threshold,  // Threshold at which we should stop splitting
        double& min_ub,  // Current minimum upper bound
        minimizer_list& ml) // List of current minimizers
{
  omp_set_num_threads(4);

  interval fxy = f(x,y);

  if (fxy.left() > min_ub) { // Current box cannot contain minimum? ---- si l'interval ne contient pas le minimum
    return ;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum? --- si l'interval contient le minimum
    /* Discarding all saved boxes whose minimum lower bound is
     * greater than the new minimum upper bound
    */
    /* Only one thread can affect a new value to the variable to avoid conflict */
    #pragma omp critical
    {
    min_ub = fxy.right();
    auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
    ml.erase(discard_begin,ml.end());
    }
  }

  // Checking whether the input box is small enough to stop searching. ----
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) {
    // We have potentially a new minimizer
    /* Only one thread can affect a new value to the variable to avoid conflict */
    #pragma omp critical
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);

  /*#pragma omp parallel
  //#pragma omp single
  { 
    #pragma omp task
    minimize(f,xl,yl,threshold,min_ub,ml);
    
    #pragma omp task
    minimize(f,xl,yr,threshold,min_ub,ml);
    
    #pragma omp task
    minimize(f,xr,yl,threshold,min_ub,ml);
    
    #pragma omp task
    minimize(f,xr,yr,threshold,min_ub,ml);
  }*/

  #pragma omp parallel
  #pragma omp sections
  {
    #pragma omp section
    minimize(f,xl,yl,threshold,min_ub,ml);
    #pragma omp section
    minimize(f,xl,yr,threshold,min_ub,ml);
    #pragma omp section
    minimize(f,xr,yl,threshold,min_ub,ml);
    #pragma omp section
    minimize(f,xr,yr,threshold,min_ub,ml);
  }
  
  /*
  minimize(f,xl,yl,threshold,min_ub,ml);
  minimize(f,xl,yr,threshold,min_ub,ml);
  minimize(f,xr,yl,threshold,min_ub,ml);
  minimize(f,xr,yr,threshold,min_ub,ml);
  */
  
}

int main(int argc, char* argv[])
{
   //Initialize MPI

  int numprocs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  clock_t debut, fin;
  cout.precision(16);
  // By default, the currently known upper bound for the minimizer is +oo

  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is 
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision;

  // Name of the function to optimize
  string choice_fun;

  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  
  bool good_choice;

  //size of each sub interval
  double szX = 0.0; 
  double szY = 0.0;

  //left bound of each interval 
  double ly = 0.0;
  double lx = 0.0;

  double min_ub = numeric_limits<double>::infinity();
  double localMin_ub = numeric_limits<double>::infinity();
  //Partager le fun.x en numprocs intervals
  interval arrayX[numprocs];
  interval arrayY[numprocs];
  //local
  interval myslice[1];

  // Asking for the threshold below which a box is not split further
  if(rank == 0){
      // Asking the user for the name of the function to optimize
    do {
      good_choice = true;

      cout << "Which function to optimize?\n";
      cout << "Possible choices: ";
      for (auto fname : functions) {
        cout << fname.first << " ";
      }
      cout << endl;
      cin >> choice_fun;
      
      try {
        fun = functions.at(choice_fun);
      } catch (out_of_range) {
        cerr << "Bad choice" << endl;
        good_choice = false;
      }
    } while(!good_choice);

    cout << "Precision? ";
    cin >> precision;
  
 

  }


  //Bcast on precision and fun
  MPI_Bcast(&precision, 1, MPI_DOUBLE, 0,MPI_COMM_WORLD );
  MPI_Bcast(&fun, sizeof(opt_fun_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  debut = clock();

  //Divide intervals
  if(rank == 0){

    //sz : size of each interval
    szX = (fun.x.width())/numprocs;
    szY = (fun.y.width())/numprocs;
    
    //Divide interval X
    lx = fun.x.left();
    //#pragma omp parallel for
    for(int i=0; i<numprocs; ++i){
      interval tmp = interval(lx,lx+szX);
      arrayX[i] = tmp;
      lx += szX;     
    }
    //Divide interval Y
    ly = fun.y.left();
    //#pragma omp parallel for
    for(int i=0; i<numprocs; ++i){
      interval tmp = interval(ly,ly+szY);
      arrayY[i] = tmp;
      ly += szY;     
    }

  }

  
  //Bcast on arrayY
  MPI_Bcast(&arrayY, numprocs * sizeof(interval), MPI_BYTE, 0, MPI_COMM_WORLD);
  //Scatter on arrayx
  MPI_Scatter(&arrayX, sizeof(interval), MPI_BYTE, &myslice, sizeof(interval),MPI_BYTE,0,MPI_COMM_WORLD);
    
  //minimize
  #pragma omp parallel for
  for(int i=0; i<numprocs ; ++i){
    //#pragma omp critical
    minimize(fun.f,myslice[0],arrayY[i],precision,localMin_ub,minimums);
  }

  //Reduction
  MPI_Reduce(&localMin_ub, &min_ub, 1, MPI_DOUBLE, MPI_MIN,0, MPI_COMM_WORLD);

  fin = clock();

  if (rank == 0) {
    // Displaying all potential minimizers
    //copy(minimums.begin(),minimums.end(),
      //   ostream_iterator<minimizer>(cout,"\n"));    
    cout << "Number of minimizers: " << minimums.size() << endl;
    cout << "Upper bound for minimum: " << min_ub << endl;
    cout << "temps : " << (double) (fin-debut)/CLOCKS_PER_SEC << "s" << endl;
  }
  MPI_Finalize();
  
}
