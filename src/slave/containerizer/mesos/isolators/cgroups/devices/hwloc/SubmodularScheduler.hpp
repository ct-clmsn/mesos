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
//  The algorithm used for this Scheduler is based
//  on an extractive summarization technique found
//  in the paper,
//
//  "Multi-document Summarization via Budgeted
//   Maximization of Submodular Functions" by
//  Hui Lin & Jeff Bilmes naaclhlt2010.pdf
//
//  This technique exploits the diminishing returns
//  property associated with submodular selection
//  algorithms.
//
//  This technique selects B cores, each with a cost
//  of 1 and with a "value" equivalent to the latency
//  of selecting a core multipled by the weighted
//  cost of work being currently performed on the
//  core being considered
//
//  By default the least expensive core will be
//  will be selected.
//
//  blame ct-clmsn (github)
//

#ifndef __MESOSSUBMODSCHEDULER__
#define __MESOSSUBMODSCHEDULER__ 1

#include "hwloc.hpp"

#include <set>
#include <algorithm>
#include <map>

using namespace std;

namespace mesos {
namespace internal {
namespace slave {

class SubmodularScheduler {
private:
  set<hwloc_obj_t> U, V;
  map<hwloc_obj_t, float> cost, weights;

  float f(TopologyResourceInformation& loc, set<hwloc_obj_t> G);

public:
  SubmodularScheduler(map<hwloc_obj_t, float> cost_map,
                      map<hwloc_obj_t, float> weight_map);

  void run(TopologyResourceInformation& info,
           set<hwloc_obj_t>& Gf,
           const float B,
           const float r = 1.0,
           const float differenceEpsilon = 0.75);
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {


#endif
