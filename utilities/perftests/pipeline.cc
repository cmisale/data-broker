#include <mpi.h>

#include <iostream>
#include <iomanip>

#include "timing.h"
#include "commandline.h"
#include "resultdata.h"
#include "requestdata.h"

#include "libdatabroker.h"
#include "benchmark.h"

#define MAX_TEST_MEMORY_USE ( 2 * 1024ull * 1024ull * 1024ull ) 

void PrintParallelResultLine( dbr::config *cfg,
                              int testcase,
                              dbr::resultdata *resd,
                              double actual_time,
                              MPI_Comm comm )
{
    if( (cfg->_testcase & testcase) == 0 )
    {
        dbr::DestroyResult( resd );
        return;
    }

    
    int pid = 0;
    int np = 0;
    MPI_Comm_size( comm, &np );
    MPI_Comm_rank( comm, &pid );
    
    int writers = np/2;
    
    
    double total_req = np * ( cfg->_iterations - cfg->_inflight - cfg->_inflight );
    double total_time = 0.0;
    if (testcase == dbr::TEST_CASE_PUTGET) {
        if (pid >= writers) {
            MPI_Send( &actual_time, 1, MPI_DOUBLE, 0, 0, comm );
        }
        if (pid == 0) {
            double tmp_time = 0;
            for (int i = writers; i < np; ++i) {
                MPI_Recv(&tmp_time, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
                total_time += tmp_time;
            }  
        }
    } else if (testcase == dbr::TEST_CASE_PUT) {
        if (pid < writers && pid > 0) {
            MPI_Send( &actual_time, 1, MPI_DOUBLE, 0, 0, comm );
        }
        if (pid == 0) {
            double tmp_time = actual_time;
            for (int i = 1; i < writers; ++i) {
                MPI_Recv(&tmp_time, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
                total_time += tmp_time;
            }  
        }
    }
    // MPI_Allreduce( &actual_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, comm );
    total_time = total_time/writers;

    double minlat = 100000000000.;
    double glob_min = 100000000000.;
    double maxlat = 0.0;
    double glob_max = 0.0;
    double total = 0.0;
    size_t count = 0;
    for( size_t n=cfg->_inflight; n<cfg->_iterations-cfg->_inflight; ++n )
    {
        minlat = std::min( minlat, resd->_latency[ n ] );
        maxlat = std::max( maxlat, resd->_latency[ n ] );
        total += resd->_latency[ n ];
        ++count;
    //    std::cout << " " << resd->_latency[ n ];
    }
    //  std::cout << "rank " << pid << " min max " << minlat << " " << maxlat << std::endl;

    


    if (testcase == dbr::TEST_CASE_PUTGET) {
        // compute min latency
        if (pid >= writers) {
            MPI_Send( &minlat, 1, MPI_DOUBLE, 0, 0, comm );
        }
        if (pid == 0) {
            double  tmp = 0.0;
            
            for (int i = writers; i < np; ++i) {
                MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
                glob_min = std::min(tmp, glob_min);
            }  
        }
        // compute max latency
        if (pid >= writers) {
            MPI_Send( &maxlat, 1, MPI_DOUBLE, 0, 0, comm );
        }
        if (pid == 0) {
            double  tmp = 0.0;
            for (int i = writers; i < np; ++i) {
                MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
                glob_max = std::max(tmp, glob_max);
            }  
        }

    } else if (testcase == dbr::TEST_CASE_PUT) {
        // compute min latency
        if (pid < writers && pid > 0) {
            MPI_Send( &minlat, 1, MPI_DOUBLE, 0, 0, comm );
        }
        if (pid == 0) {
            double  tmp = 0.0;
            glob_min = minlat;
            for (int i = 1; i < writers; ++i) {
                MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
                glob_min = std::min(tmp, glob_min);
            }  
        }
        // compute max latency
        if (pid < writers && pid > 0) {
            MPI_Send( &maxlat, 1, MPI_DOUBLE, 0, 0, comm );
        }
        if (pid == 0) {
            double  tmp = 0.0;
            glob_max = maxlat;
            for (int i = 1; i < writers; ++i) {
                MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
                glob_max = std::max(tmp, glob_max);
            }  
        }
    }
    // MPI_Allreduce( &minlat, &glob_min, 1, MPI_DOUBLE, MPI_MIN, comm );
    // MPI_Allreduce( &maxlat, &glob_max, 1, MPI_DOUBLE, MPI_MAX, comm );

    if( pid == 0 )
    {
        std::cout << std::setw(10) << cfg->_datasize
            << std::setw(12) << total_time/1000.
            << std::setw(12) << total_req
            << std::setw(12) << (total_req)/(total_time/1000000.)
            << std::setw(12) << (total_req*cfg->_datasize)/total_time
            << std::setw(12) << (total_time*np/1000./total_req)
            << std::setw(12) << glob_min/1000.
            << std::setw(12) << glob_max/1000.
            << std::setw(12) << dbr::case_str[ testcase ]
            << std::endl;
    }
    MPI_Barrier( comm );
    dbr::DestroyResult( resd );
}


int main (int argc, char** argv) {
    int np,pid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&pid);

    std::string extraHelp = "\
    -p <inflight>      number of requests that are kept in-flight at the same time (1)\n";

    dbr::config *config = dbr::ParseCommandline( argc, argv, "d:hk:Kn:p:", dbr::par_single_common::extraParse, extraHelp, true );
    if( config == NULL )
    {
        std::cerr << "Failed to create configuration." << std::endl;
        return -1;
    }

    dbr::test_start = dbr::myTime();

    dbr::resultdata *put_res = dbr::InitializeResult( config );
    dbr::resultdata *getb_res = dbr::InitializeResult( config );

    dbr::requestdata *reqd = dbr::InitializeRequest( config );

    char *data = dbr::generateLongMsg( config->_datasize );

    if( config->_iterations * config->_keylen < MAX_TEST_MEMORY_USE ) {
        for( size_t n=0; n<config->_iterations; ++n )
        dbr::RandomizeData( reqd, n, (config->_variable_key * random() % config->_keylen ) + config->_keylen );
    }

    DBR_Handle_t h;
    if( pid == 0 ) {
        h = dbrCreate((DBR_Name_t)TEST_NAMESPACE, DBR_PERST_VOLATILE_SIMPLE, DBR_GROUP_LIST_EMPTY );
        MPI_Barrier(comm);
    } else {
        MPI_Barrier(comm);
        h = dbrAttach( (DBR_Name_t)TEST_NAMESPACE );
    }
    if( h == NULL ) {
        std::cerr << "Failed to create namespace" << std::endl;
        exit( -1 );
    }

    double put_actual_time = 0.0;
    double getb_actual_time = 0.0;
    bool first = true;

    int writers = np/2;
    if( pid == 0 ) { std::cout << "."; std::flush( std::cout ); }
    if (pid < writers) {
        if(first) {
           std::cout << "I am pid " << pid << " performing PUT" << std::endl;
           std::flush(std::cout);
           first = false;
        }
       put_actual_time = dbr::benchmark(config, dbr::TEST_CASE_PUT, put_res, reqd, h, data );
    } else {
        if(first) {
           std::cout << "I am pid " << pid << " performing GET" << std::endl;
           std::flush(std::cout);
           first = false;
        }
       getb_actual_time = dbr::benchmark(config, dbr::TEST_CASE_GETB, getb_res, reqd, h, data );
    }
  if( pid == 0 ) { std::cout << "."; std::flush( std::cout ); }
   MPI_Barrier(comm);

    if( pid == 0 )
  {
    MPI_Barrier(comm);
    dbrDelete( (DBR_Name_t)TEST_NAMESPACE );
  }
  else
  {
    dbrDetach( h );
    MPI_Barrier(comm);
  }

  if( pid == 0 ) std::cout << "Done." << std::endl;

  if( pid == 0 )
    dbr::PrintHeader( config, argc, argv );

  PrintParallelResultLine( config, dbr::TEST_CASE_PUT, put_res, put_actual_time, comm );
  PrintParallelResultLine( config, dbr::TEST_CASE_PUTGET, getb_res, getb_actual_time, comm );

  dbr::DestroyRequest( reqd );
  delete config;

  MPI_Finalize();
    return 0;
}