/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

__global__ void kernel2(int MIN,int level,int V,int L,int *d_max,int *d_offset,int *d_csrList,int *d_apr,int* d_aid,int* d_activeVertex){
    __shared__ int count;//local count of a block
    __shared__ int s_max;//local max of a block

    int MAX;
    
    MAX = d_max[blockIdx.x];
    int thid = MIN + blockIdx.x*blockDim.x + threadIdx.x;
    
    int start,end;
    
    int th_max;

    th_max = -1;

    s_max = -1;

    __syncthreads();

    //condition to check valid thid
    if(thid<=MAX && thid<V){
        start = d_offset[thid];//start index of csrlist
        end = d_offset[thid+1];//end index of csrlist

        //condition ot heck if a thread is active. Active Threads increase aid of their adjacent nodes by 1.
        if(d_aid[thid]>=d_apr[thid]){
            if(thid==MIN || thid==MAX ||(thid!=MIN && thid!=MAX && (d_aid[thid-1]>=d_apr[thid-1] || d_aid[thid+1]>=d_apr[thid+1]) ) ){
                atomicAdd(&count,1);//add 1 to shared count of the block
                for(int i=start;i<end;i++){
                    atomicAdd(&d_aid[d_csrList[i]],1);
                }
            }
        }
        //All the  thread of a level compute locally the max vertex id they are adjacent to in next level.
        for(int i=start;i<end;i++){
            int temp1;
            temp1 = d_csrList[i];
            //if(th_min>temp1) th_min = temp1;
            if(th_max<temp1) th_max = temp1;
        }

        //All the threads of block compute max vertex adjacnet in next level by threads in a block.
        atomicMax(&s_max,th_max);
        
        //By now s_max contains max vertex adjacent in next level by threads in a block and active threads have increased adjacent vertex aid by 1. 
        // Count contains number of active vertices in a block for current particular level.
        __syncthreads();
        if(threadIdx.x==0){
            
            //Store max vertex of a block to its corresponding position in d_max.
            d_max[blockIdx.x]=s_max;

            //Add count to d_activeVertex of the current level.
            atomicAdd(&d_activeVertex[level],count);
            count=0;
        }
    }

}

    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

int *d_activeVertex;// Store number of active vertex for all level.
cudaMalloc(&d_activeVertex, L*sizeof(int));
//Initialize number of active vertex for all levels as 0.
cudaMemset(d_activeVertex,0,L*sizeof(int));
//Initialize aid for all vertices as 0.
cudaMemset(d_aid,0,V*sizeof(int));


//d_max to store max reachable vertex of all the vertices in a block for a level.
//h_max cpu copy of d_max to compute max reachble vertex of next level.
//temp_max to intilize d_max before each krnel call for a level.
int *d_max,*h_max,*temp_max;

//min to store first vertex of a level.
//max to store last vertex of a level.
int min,max;
min=0;
max=-1;
//Compute last vertex of level 1.
for(int i=0;i<V;i++){
    if(h_apr[i]!=0) break;
    max++;
}
int num_block,vertex;
//calculate kerenel lauch parameters.
vertex = max+1;
num_block = ceil((float)vertex/1024);
h_max = (int *)malloc(10 * sizeof(int));
temp_max = (int *)malloc(10 * sizeof(int));

memset(h_max,-1,10*sizeof(int));

cudaMalloc(&d_max,10*sizeof(int));

//cudaMemcpy(d_max,&max,sizeof(int),cudaMemcpyHostToDevice);

dim3 blocksize(1024,1,1);

//temp variable to store last index of level for next iteration.
int t_max=max;
for(int i=0;i<L;i++){
    
    t_max=-1;
    //caculate lauch parameters for kernle launch.
    num_block = ceil((float)vertex/1024);
    
    //initialize temp_max
    memset(temp_max,-1,10*sizeof(int));
    
    //intialize d_max
    for(int j=0;j<num_block;j++){
        temp_max[j]=max;    
    }
    cudaMemcpy(d_max,temp_max,10 * sizeof(int),cudaMemcpyHostToDevice);
    
    
    dim3 gridsize(num_block,1,1);
    // kernel lauch of a level.
    kernel2<<<gridsize,blocksize>>>(min,i, V, L, d_max, d_offset, d_csrList, d_apr, d_aid, d_activeVertex);
    
    //Copy d_max to h_max and calcuate last vertex of next level. 
    cudaMemcpy(h_max,d_max,10 * sizeof(int),cudaMemcpyDeviceToHost);
    for(int j=0;j<num_block;j++){
        
        if(h_max[j]>t_max) t_max = h_max[j];
    }
    //update first and last index of next level.
    min = max+1;
    max = t_max;
    //reset h_max 
    memset(h_max,-1,10*sizeof(int));
    
    //calculate threads needed for next level.
    vertex = max-min+1;
}


/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host

cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);

char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
