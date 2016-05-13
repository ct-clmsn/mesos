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

#include "ProcessBinder.hpp"
#include "SubmodularScheduler.hpp"

#include "stout/foreach.hpp"

#include <vector>
#include <map>
#include <set>

using namespace std;

namespace mesos {
namespace internal {
namespace slave {

#ifdef USE_CUDA

Try<bool> FALSE_ON_FAIL(CUresult& res)
{
  if (CUDA_SUCCESS != res) {
    error(-1, 1, "cuInit returned error");
    return false;
  }

  return true;
}

#endif

static void updateCoreUtilizationHistogram(TopologyResourceInformation& loc,
                                           pid_t pid,
                                           hwloc_obj_t core)
{
  map< hwloc_obj_t, float >::iterator j =
           loc.coreUtilizationHistogram.find(core);

  map< pid_t, vector<hwloc_obj_t> >::iterator pidcore =
           loc.pidCoreUtilization.find(pid);

  if(pidcore == loc.pidCoreUtilization.end()) {
    vector<hwloc_obj_t> vec;
    vec.push_back(core);
    loc.pidCoreUtilization.insert(make_pair(pid, vec));
  }
  else {
    pidcore->second.push_back(core);
  }

  if(j == loc.coreUtilizationHistogram.end()) {
    loc.coreUtilizationHistogram.insert(make_pair(core, 1.0));
  }
  else {
    loc.coreUtilizationHistogram[core] =
        loc.coreUtilizationHistogram[core]+1.0;
  }
}

Try<bool> pin_executor_cpu(TopologyResourceInformation& loc,
                           const pid_t pid,
                           const string cgroup,
                           const float ncpus_req,
                           const float ngpus_req)
{
  const float npus = (float)loc.nProcessingUnitsPerMachine();

  // percentage of available processing units on the machine
  const uint32_t core_pus_req = (npus > 0) ?
                                (uint32_t)(lround((npus *
                                ncpus_req))) : 0;

  std::map<hwloc_obj_t, float> cost_vec;
  foreachpair(hwloc_obj_t core, std::vector<hwloc_obj_t> pu_set,
              loc.pusPerCore) {
    // each core "cost" is equal to the number of processing units available
    cost_vec.insert(make_pair(core, (float)pu_set.size()));
  }

#ifdef USE_CUDA

  const bool pid_req_gpu = ngpus_req.isNone() ? false : true;

  // gpus required! ignore histogram
  // utilization model - priority
  // is on packing work closest to
  // the gpu(s)
  if(pid_req_gpu)
  {
    // task asked for a gpu, hwloc
    // didn't find a gpu
    // "houston, we have a problem!"
    if(loc.grpus.size() < 1) {
      error(-1, 1, "task requires gpu hwloc found no gpus");
      return false;
    }

    // task asked for a gpu, cuda
    // didn't initialize properly
    // "houston, we have a problem!"
    if(FALSE_ON_FAIL(cuInit(0))) {
      error(-1, 1, "task requires gpu cuda did not initialize");
      return false;
    }

    uint32_t total_pus_req = core_pus_req;

    const uint32_t gpus_req =
         (uint32_t)std::floor(((float)loc.gpus.size()) * ngpus_req.get());

    hwloc_cpuset_t pid_binding_cpuset = hwloc_bitmap_alloc();

    for(int i = 0; i < gpus_req; i++) {
      hwloc_obj_t gpu = loc.gpus[i];
      char pciBusId[16];
      char devName[256];
      CUdevice dev;

      sprintf(pciBusId,
              "%.2x:%.2x:%.2x.%x",
              gpu->attr->pcidev.domain,
              gpu->attr->pcidev.bus,
              gpu->attr->pcidev.dev,
              gpu->attr->pcidev.func);

      if(FALSE_ON_FAIL(cuDeviceGetByPCIBusId(&dev, pciBusId))) {
        error(-1, 1, "task requires gpu cuda did not find the pcibusid");
        return false;
      }
      if(FALSE_ON_FAIL(cuDeviceGetName(devName, 256, dev))) {
        error(-1, 1, "task requires gpu cuda did not find the device id");
        return false;
      }

      hwloc_cpuset_t gpu_associated_cpuset = hwloc_bitmap_alloc();

      // https://www.open-mpi.org/projects/hwloc/doc/v1.9.1/a00107.php
      //
      // get the CPU set of logical processors that are physically
      // close to device cudevice.
      hwloc_cuda_get_device_cpuset(loc.topology,
                                   gpu,
                                   dev,
                                   gpu_associated_cpuset);

      hwloc_bitmap_or(pid_binding_cpuset,
                      pid_binding_cpuset,
                      gpu_associated_cpuset);

      hwloc_bitmap_free(gpu_associated_cpuset);
    }

    vector<hwloc_obj_t> used_cpuset, remaining_cpuset;

    // collect information about the cpus selected as "closest to gpus"
    // collect information about the cpus not-selected
    foreachkey(hwloc_obj_t cur_core, loc.pusPerCore) {
      if(hwloc_bitmap_isincluded(pid_binding_cpuset,
                                 cur_core->cpuset)) {
        used_cpuset.push_back(cur_core);

        // figure out the total number of processing units still required
        const std::map<hwloc_obj_t, float>::iterator core_pucount =
            cost_vec.find(cur_core);

        total_pus_req -= (core_pu_count == cost_vec.end()) ?
            0 : (uint32_t)core_pucount->second;

        // updates tasks counts on cores nearest the
        // gpu on this locale
        updateCoreUtilizationHistogram(loc, pid, cur_core);
      }
      else {
        remaining_cpuset.insert(cur_core);
      }
    }

    // if after cpu-gpu selection if cores are still needed
    if(total_pus_req > 0) {
      // make sure there are cores left over
      if(used_cpuset.size() > 0 && remaining_cpuset.size() > 0) {
        // grab the first cpu found to be near a gpu
        const hwloc_obj_t core = *(used_cpuset.begin());
        const uint32_t core_idx = core->logical_index;

        vector< pair<hwloc_obj_t, float> > ordered_remaining;

        // find the distances of cpus remaining that are
        // "near" the first cpu by a gpu
        foreach(hwloc_obj_t remaining_coreset, remaining_cpuset) {
          const float dist =
              get_core_distance(loc, core_idx, remaining_coreset);
          ordered_remaining.push_back(make_pair(remaining_coreset, dist));
        }

        // sort those 2nd degree cpus by distance
        sort(ordered_remaining.begin(),
             ordered_remaining.end(),
             sort_pred);

        // cycle through the sorted list and add these left-over
        // cpus to the cpuset
        foreach(pair<hwloc_obj_t, float> k, ordered_remaining) {
          const hwloc_obj_t core = k->first;

          hwloc_bitmap_or(pid_binding_cpuset,
                          pid_binding_cpuset,
                          core->cpuset);

          // this update is to capture remaining
          // cores used for gpu compute
          updateCoreUtilizationHistogram(loc, pid, core);

          // figure out the total number of processing units still required
          const std::map<hwloc_obj_t, float>::iterator core_pucount =
              cost_vec.find(core);

          total_pus_req -= (core_pu_count == cost_vec.end()) ?
              0.0 : (uint32_t)core_pucount->second;

          vector<hwloc_obj_t>::iterator remaining_core =
              remaining_cpuset.find(core);

          // in remove the cpus from the remaining set
          // this is done to make the previous
          // remaining_cpuset.find work correctly
          if(remaining_core != remaining_cpuset.end()) {
            remaining_cpuset.erase(coreitr);
          }
        }
      }
    }

    hwloc_set_proc_cpubind(loc.topology,
                           pid,
                           pid_binding_cpuset,
                           HWLOC_CPUBIND_PROCESS);

    hwloc_bitmap_free(pid_binding_cpuset);
    hwloc_topology_destroy(loc.topology);
  }
  else {
#endif

    hwloc_cpuset_t pid_binding_cpuset = hwloc_bitmap_alloc();

    // if task count == total number of cores, randomly assign
    // means the task queue is empty per core
    //
    if(loc.getTaskCount() == ((float)loc.nCoresPerMachine()))  {
      vector<hwloc_obj_t> cores;
      loc.randCpuAssigner(cores, core_pus_req);

      foreach(hwloc_obj_t core, cores) {
        hwloc_bitmap_or(pid_binding_cpuset,
                       pid_binding_cpuset,
                       core->cpuset);

        updateCoreUtilizationHistogram(loc, pid, core);
      }
    }
    // tasks have been assigned, use
    // greedy submodular selection
    //
    else {
      map<hwloc_obj_t, float> weight_vec;
      // get task weights - #tasks-on-a-core / #core-processing-units
      loc.getWeightedTaskFrequencyVector(weight_vec);

      SubmodularScheduler scheduler(cost_vec, weight_vec);
      set<hwloc_obj_t> G;
      scheduler.run(loc, G, core_pus_req);

      foreach(hwloc_obj_t core, G) {
        hwloc_bitmap_or(pid_binding_cpuset,
                        pid_binding_cpuset,
                        core->cpuset);

        updateCoreUtilizationHistogram(loc, pid, core);
      }
    }

    hwloc_set_proc_cpubind(loc.topology,
                           pid,
                           pid_binding_cpuset,
                           HWLOC_CPUBIND_PROCESS);

    hwloc_bitmap_free(pid_binding_cpuset);
    hwloc_topology_destroy(loc.topology);

  #ifdef USE_CUDA
  }

  #endif

  return true;
}

Try<bool> unpin_executor_cpu(TopologyResourceInformation& loc,
                             const pid_t pid)
{
  map<pid_t, vector<hwloc_obj_t> >::iterator hist =
      loc.pidCoreUtilization.find(pid);

  if(hist == loc.pidCoreUtilization.end()) { return false; }

  foreach(hwloc_obj_t core, hist->second) {
    loc.coreUtilizationHistogram[core] =
        loc.coreUtilizationHistogram[core] - 1.0;
  }

  loc.pidCoreUtilization.erase(pid);

  return true;
}


} // namespace slave {
} // namespace internal {
} // namespace mesos {
