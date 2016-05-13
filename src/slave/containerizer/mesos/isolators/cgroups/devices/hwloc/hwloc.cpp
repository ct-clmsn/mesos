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
//  blame ct-clmsn (github)
//

#include "stout/foreach.hpp"

#include "slave/containerizer/mesos/isolators/cgroups/cpushare.hpp"

#include "hwloc.hpp"

#include <error.h>
#include <utility>
#include <algorithm>
#include <limits>

namespace mesos {
namespace internal {
namespace slave {

float TopologyResourceInformation::getCoreDistance(const int i, const int j) {
  return coreDistmat[i*nCoresPerMachine()+j].latency[i*nCoresPerMachine()+j];
}

struct sortpred {
  bool operator()(const std::pair<int, float> &left,
                  const std::pair<int, float> &right) {
    return left.second < right.second;
  }
} sort_pred;

inline int map_acc(float lhs, const std::pair<hwloc_obj_t, float> & rhs) {
  return lhs + rhs.second;
}

TopologyResourceInformation::TopologyResourceInformation() {
  if (hwloc_topology_init(&(topology))) {
    /* error in initialize hwloc library */
    error(-1, 1, "%s: hwloc_loc.topo_init() failed", __func__);
  }

  const unsigned long topo_flags = HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM |
                                   HWLOC_TOPOLOGY_FLAG_IO_DEVICES |
                                   HWLOC_TOPOLOGY_FLAG_IO_BRIDGES;

  hwloc_topology_set_flags(topology, topo_flags);
  hwloc_topology_load(topology);

  root = hwloc_get_root_obj(topology);

  discoverCpuTopology(topology, root, NULL);
  coreDistmat = hwloc_get_whole_distance_matrix_by_type(topology,
                                                      HWLOC_OBJ_CORE);

  discoverGpuTopology(topology, root, NULL);

  foreachkey(hwloc_obj_t core, pusPerCore) {
    // each core needs a minimal "utilization"
    // value to make the (latency * utilization)
    // trade off work as expected
    //
    coreUtilizationHistogram.insert(make_pair(core, 1.0));
  }
}

int TopologyResourceInformation::nSocketsPerMachine() {
  return coresPerSocket.size();
}

int TopologyResourceInformation::nCoresPerMachine() {
  return pusPerCore.size();
}

int TopologyResourceInformation::nProcessingUnitsPerMachine() {
  std::vector<int> pu_vec;
  nProcessUnitsPerCore(pu_vec);
  return std::accumulate(pu_vec.begin(), pu_vec.end(), 0);
}

void TopologyResourceInformation::nCoresPerSocket(
    std::vector<int>& coreCounts )
{
  foreachvalue(std::vector<hwloc_obj_t> socket_cores, coresPerSocket) {
    coreCounts.push_back(socket_cores.size());
  }
}

int TopologyResourceInformation::getCoreIndex(hwloc_obj_t core) {
  int count = 0;
  foreachvalue(std::vector<hwloc_obj_t> socket_cores, coresPerSocket) {
    vector<hwloc_obj_t>::iterator i =
        find(socket_cores.begin(), socket_cores.end(), core);
    if(i == socket_cores.end()) {  return -1; }
    count+=1;
  }

  return count;
}

void TopologyResourceInformation::nProcessUnitsPerCore(
    std::vector<int>& puCounts )
{
    foreachvalue(std::vector<hwloc_obj_t> core_pu, pusPerCore) {
        puCounts.push_back(core_pu.size());
    }
}

void TopologyResourceInformation::getCpusetsPerCore(
    std::vector<hwloc_cpuset_t> cpusetsVec )
{
  foreachkey(hwloc_obj_t core, pusPerCore) {
     cpusetsVec.push_back(hwloc_bitmap_dup(core->allowed_cpuset));
  }
}

float TopologyResourceInformation::getTaskCount() {
  const float sum = std::accumulate(coreUtilizationHistogram.begin(),
                                    coreUtilizationHistogram.end(),
                                    0,
                                    map_acc);
  return sum;
}

void TopologyResourceInformation::getTaskFrequencyVector(
    std::vector<float>& costVec )
{
  foreachpair(hwloc_obj_t core,
              float cost,
              coreUtilizationHistogram)
  {
     std::map< hwloc_obj_t,
               std::vector<hwloc_obj_t> >::iterator puCounts =
                   pusPerCore.find(core);

     costVec.push_back( (puCounts != pusPerCore.end()) ?
                         cost :
                         numeric_limits<float>::infinity() );
  }
}

void TopologyResourceInformation::getWeightedTaskFrequencyVector(
    std::vector<float>& costVec )
{
  foreachpair(hwloc_obj_t core,
              float cost,
              coreUtilizationHistogram)
  {
    std::map< hwloc_obj_t,
              std::vector<hwloc_obj_t> >::iterator puCounts =
                   pusPerCore.find(core);

    costVec.push_back( (puCounts != pusPerCore.end()) ?
                        ( ((float)puCounts->second.size()) / cost ) :
                        - numeric_limits<float>::infinity());
  }
}

void TopologyResourceInformation::getWeightedTaskFrequencyVector(
    std::map<hwloc_obj_t, float>& costVec)
{
  foreachpair(hwloc_obj_t core,
              float cost,
              coreUtilizationHistogram)
  {
    std::map< hwloc_obj_t,
              std::vector<hwloc_obj_t> >::iterator puCounts =
                  pusPerCore.find(core);

    costVec.insert( make_pair( core, (puCounts != pusPerCore.end()) ?
                         ( ((float)puCounts->second.size()) / cost ) :
                         - numeric_limits<float>::infinity()) );
  }
}

static bool find_parent_by_type(
  hwloc_obj_t halt,
  hwloc_obj_t obj,
  const hwloc_obj_type_t T)
{
  for(hwloc_obj_t cur = obj; cur != NULL; cur = cur->parent) {
    if(cur->type == T) { return true; }
  }

  return false;
}

void TopologyResourceInformation::discoverCpuTopology(hwloc_topology_t topology,
                                                      hwloc_obj_t parent,
                                                      hwloc_obj_t child) {
    hwloc_obj_t component;
    component = hwloc_get_next_child(topology, parent, child);

    if (NULL == component) {
        return;
    }

    // add vector of cores to socket map
    if (component->type == HWLOC_OBJ_SOCKET) {
       std::vector<hwloc_obj_t> cores_vec;
       coresPerSocket.insert(std::make_pair(component, cores_vec));
    }

    // add core to socket
    else if (component->type == HWLOC_OBJ_CORE) {
      foreachpair(hwloc_obj_t socket,
                  vector<hwloc_obj_t> cores,
                  coresPerSocket)
      {
        if(find_parent_by_type(socket, component, HWLOC_OBJ_SOCKET)) {
          cores.push_back(component);
          coresPerSocket.insert(std::make_pair(socket, cores));
        }
      }
    }

    // add pu to core
    else if (component->type == HWLOC_OBJ_PU) {
        if(find_parent_by_type(parent, component, HWLOC_OBJ_CORE)) {
            std::map< hwloc_obj_t,
                std::vector<hwloc_obj_t> >::iterator pu_core =
                    pusPerCore.find(parent);

            if(pu_core == pusPerCore.end()) {
                std::vector<hwloc_obj_t> pusvec;
                pusvec.push_back(component);
                pusPerCore.insert(std::make_pair(parent, pusvec));
            }
            else {
                pu_core->second.push_back(component);
            }
        }
    }

    if (0 != component->arity) {
        /* This device has children so need to look recursively at them */
        discoverCpuTopology(topology, component, NULL);
        discoverCpuTopology(topology, parent, component);
    }
    else {
        discoverCpuTopology(topology, parent, component);
    }
}

// modified from http://icl.cs.utk.edu/open-mpi/faq/?category=runcuda
//
void TopologyResourceInformation::find_gpus(
  hwloc_obj_t parent,
  hwloc_obj_t child)
{
  hwloc_obj_t pcidev;
  pcidev = hwloc_get_next_child(topology, parent, child);

  if (NULL == pcidev) {
    return;
  }
  else if (0 != pcidev->arity) {
    find_gpus(pcidev, NULL);
    find_gpus(parent, pcidev);
  }
  else {
    if (pcidev->attr->pcidev.vendor_id == 0x10de) {
      gpus.push_back(pcidev);
    }

    find_gpus(parent, pcidev);
  }
}

void TopologyResourceInformation::discoverGpuTopology(
  hwloc_topology_t topology,
  hwloc_obj_t parent,
  hwloc_obj_t child)
{
  hwloc_obj_t bridge;
  bridge = hwloc_get_obj_by_type(topology, HWLOC_OBJ_BRIDGE, 0);
  find_gpus(bridge, NULL);
}

static int random_core_selection(const uint32_t total_number_of_cores) {
  /* randomly die-rolls a core from the total number available */
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, total_number_of_cores);
  // generates number in the range 1..6
  int dice_roll = distribution(generator);
  return dice_roll;
}

void TopologyResourceInformation::randCpuAssigner(
    std::vector<hwloc_obj_t>& cores,
    const uint32_t corePuReq)
{
  /* randomly selects a core uses the hwloc distance matrix between
    cores find "nearest neighbors" to use for pid assignment  */
  const uint32_t ncores = nCoresPerMachine();
  const uint32_t starting_core = random_core_selection(ncores);
  uint32_t core_pu_counter = corePuReq;

  std::vector<hwloc_obj_t> hwloc_core_objs;
  foreachkey(hwloc_obj_t core, pusPerCore) {
    hwloc_core_objs.push_back(core);
  }

  std::vector< std::pair<int, float> > core_distances;

  for(uint32_t i = starting_core; i < ncores; i++) {
    const float core_distance = getCoreDistance(starting_core, i);
    core_distances.push_back(std::make_pair(i, core_distance));
  }

  std::sort(core_distances.begin(), core_distances.end(), sort_pred);

  for(uint32_t i = starting_core; i < ncores; i++) {
    if(core_pu_counter > 0) {
      cores.push_back(hwloc_core_objs[i]);
      const std::map<hwloc_obj_t, std::vector<hwloc_obj_t>>::iterator
          pu_itr = pusPerCore.find(hwloc_core_objs[i]);

      core_pu_counter -= pu_itr->second.size();
    }
    else {
      break;
    }
  }
}

} // namespace slave {
} // namespace internal {
} // namespace mesos {
