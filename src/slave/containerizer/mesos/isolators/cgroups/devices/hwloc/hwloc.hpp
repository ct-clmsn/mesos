// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
//   hwloc.hpp
//
//   provides a function to pin/bind a pid_t associated
//   with a ContainerID to an hwloc cpuset
//
//   uses resource information about the compute node,
//   provided by hwloc, to inform cpuisolator the total
//   number of processing units (PUs) provided by all
//   cores on the compute node
//
//   using the cpu resource % associated with the ContainerID
//   determine the number of cores requested
//
//   select a set of cores and pin/bind the pid_t
//   associated with a ContainerID.
//
//   1) when coreUtilizationHistogram == number of cores,
//   randomly selecting 1 core and sequentially select
//   other cores to bind the ContainerID's pid_t
//
//   2) when coreUtilizationHistogram is > number of cores,
//   use a submodular selection algorithm (greedy algorithm
//   with knapsack constraint) that picks the required number
//   of cores for a task by evaluating the latency between
//   cores and the "per-processing-unit-work" done by a core:
//   (core-histogram-value/sum-of-histogram)/core's-processing-units
//
//   3) in the case that gpu support is required for a task,
//   the gpu percentage value is interpreted as percentage
//   taken from the total number of gpus available
//
//   this code finds uses hwloc to find the number of gpus
//   on the system. then it uses hwloc to searches the system
//   bus to identify cores that are closest to the gpu. the
//   task pid_t is then pinned/bound to those cores. remaining
//   cpu resources are then selected in sequential order from
//   the last cpu associated with the last gpu
//
//   reviewed linux kernel, hwloc, and slurm documentation/source code
//
//   https://www.kernel.org/doc/Documentation/cgroup-v1/
//   https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt
//   https://www.kernel.org/doc/Documentation/cgroup-v1/cpusets.txt
//   https://www.kernel.org/doc/Documentation/cgroup-v1/cpuacct.txt
//
//   https://www.open-mpi.org/projects/hwloc/tutorials/20150921-EuroMPI-hwloc-tutorial.pdf
//   cons_res%20.pdf
//   SUG-2012-Cgroup%20.pdf
//   EECS-2012-20.pdf
//   RR-8859.pdf
//
//   https://github.com/SchedMD/slurm/tree/master/src/slurmd/common
//   https://github.com/SchedMD/slurm/blob/master/src/slurmd/common/xcgroup.h
//   https://github.com/SchedMD/slurm/blob/master/src/slurmd/common/xcpuinfo.c
//   https://github.com/SchedMD/slurm/blob/master/src/slurmd/common/xcgroup.c
//
//   https://github.com/SchedMD/slurm/tree/master/src/plugins/task/cgroup
//   https://github.com/SchedMD/slurm/blob/master/src/plugins/task/cgroup/task_cgroup.c
//   https://github.com/SchedMD/slurm/blob/master/src/plugins/task/cgroup/task_cgroup_cpuset.h
//   https://github.com/SchedMD/slurm/blob/master/src/plugins/task/cgroup/task_cgroup_cpuset.c
//   https://github.com/SchedMD/slurm/blob/master/src/plugins/task/cgroup/task_cgroup_devices.c
//
//   additional literature
//
//   http://www.glennklockwood.com/hpc-howtos/process-affinity.html
//   http://redhatstackblog.redhat.com/2015/05/05/cpu-pinning-and-numa-topology-awareness-in-openstack-compute/
//   http://vanillajava.blogspot.com/2013/07/micro-jitter-busy-waiting-and-binding.html
//   http://vanillajava.blogspot.fr/2012/02/how-much-difference-can-thread-affinity.html
//   http://vanillajava.blogspot.com/2011/12/thread-affinity-library-for-java.html
//   http://highscalability.com/blog/2012/3/29/strategy-exploit-processor-affinity-for-high-and-predictable.html
//   http://highscalability.com/blog/2013/10/23/strategy-use-linux-taskset-to-pin-processes-or-let-the-os-sc.html
//
//  blame ct-clmsn (github)
//

#ifndef __MESOSHWLOC__
#define __MESOSHWLOC__ 1

#include <vector>
#include <map>

#include <stout/try.hpp>
#include <stout/option.hpp>

#include <hwloc.h>

using namespace std;


namespace mesos {
namespace internal {
namespace slave {

class TopologyResourceInformation {
private:
  void discoverGpuTopology(hwloc_topology_t topology,
                           hwloc_obj_t parent,
                           hwloc_obj_t child);

  void discoverCpuTopology(hwloc_topology_t topology,
                           hwloc_obj_t parent,
                           hwloc_obj_t child);

  void find_gpus(hwloc_obj_t parent,
                 hwloc_obj_t child);

public:
  // constructor, inits
  // several member variables
  // detects hwloc support
  // finds all relevant devices
  // system components
  TopologyResourceInformation();

  // get the number of sockets
  // hwloc found
  int nSocketsPerMachine();

  // get the number of cores
  // hwloc found
  int nCoresPerMachine();

  // get the number of processing
  // units hwloc found
  int nProcessingUnitsPerMachine();

  // get a list of # cores
  // per socket
  void nCoresPerSocket(std::vector<int>& coreCounts);

  // get a list of # processing
  // units per core
  void nProcessUnitsPerCore(std::vector<int>& puCounts);

  // get all the cores
  void getCpusetsPerCore(std::vector<hwloc_cpuset_t> cpusetsVec);

  // get the number of tasks
  // assigned to cores results
  // are redundant
  float getTaskCount();

  // get normalized frequency
  // of "work" per core
  void getTaskFrequencyVector(std::vector<float>& costVec);

  // get a weighted frequency
  // of "work" per core's
  // processing unit count
  void getWeightedTaskFrequencyVector(std::vector<float>& costVec);
  void getWeightedTaskFrequencyVector(std::map<hwloc_obj_t, float>& costVec);


  // randomly select a core and
  // the core's nearest neighbors
  // cores.size() == coresLen
  void randCpuAssigner(std::vector<hwloc_obj_t>& cores,
                       const uint32_t coresLen);

  // stores detected and loaded
  // hwloc topology
  hwloc_topology_t topology;

  // stores root (system/machine)
  // node in the hwloc topology
  hwloc_obj_t root;

  // keys are sockets, values are cores
  std::map< hwloc_obj_t, std::vector<hwloc_obj_t> > coresPerSocket;

  // keys are cores, values are processing units
  std::map< hwloc_obj_t, std::vector<hwloc_obj_t> > pusPerCore;

  int getCoreIndex(hwloc_obj_t core);

  // all detected gpus
  std::vector<hwloc_obj_t> gpus;

  // stores histogram of tasks to cores
  // one task could be assigned to multiple
  // cores
  std::map< hwloc_obj_t, float > coreUtilizationHistogram;
  std::map< pid_t, vector<hwloc_obj_t> > pidCoreUtilization;

  float getCoreDistance(const int i,
                        const int j);

  // stores a distance matrix (latency)
  // between all cores on the system
  const struct hwloc_distances_s* coreDistmat;
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif
