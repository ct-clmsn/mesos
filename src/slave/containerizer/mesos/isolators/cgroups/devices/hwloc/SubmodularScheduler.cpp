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
#include "SubmodularScheduler.hpp"

#include <algorithm>
#include <limits>
#include <vector>
#include <iterator>
#include <cmath>

#include "stout/foreach.hpp"

namespace mesos {
namespace internal {
namespace slave {

SubmodularScheduler::SubmodularScheduler(map<hwloc_obj_t, float> cost_map,
                                         map<hwloc_obj_t, float> weight_map) :
  cost(cost_map),
  weights(weight_map)
{
  foreachkey(hwloc_obj_t core, cost_map) {
    U.insert(core);
    V.insert(core);
  }
}

float SubmodularScheduler::f(
    TopologyResourceInformation& info,
    set<hwloc_obj_t> S )
{
  float fscore = 0.0;
  vector<hwloc_obj_t> diff;

  set_difference(V.begin(), V.end(), S.begin(), S.end(),
                 inserter(diff, diff.begin()));

  foreach(hwloc_obj_t i_itr, diff) {
    map<hwloc_obj_t, float>::iterator i_weight = weights.find(i_itr);
    const int core_i = info.getCoreIndex(i_itr);

    foreach(hwloc_obj_t j_itr, S) {
      map<hwloc_obj_t, float>::iterator j_weight = weights.find(j_itr);
      const int core_j = info.getCoreIndex(j_itr);
      float latency = info.getCoreDistance(core_i, core_j);
      latency =  (FP_ZERO == fpclassify(latency)) ? 1e-10 : latency;
      fscore += ( (i_weight->second + j_weight->second) / latency );
    }
  }

  return fscore;
}

static bool pair_cmp(pair<hwloc_obj_t, float> a,
                     pair<hwloc_obj_t, float> b) {
  return (a.second < b.second);
}

static bool fin_cmp( pair<set<hwloc_obj_t>, float> a,
                     pair<set<hwloc_obj_t>, float> b) {
  return a.second < b.second;
}

static float sum_func(float a, pair<hwloc_obj_t, float> b) {
  return a + b.second;
}

void SubmodularScheduler::run(TopologyResourceInformation& info,
                              set<hwloc_obj_t>& Gf,
                              const float B,
                              const float r,
                              const float differenceEpsilon)
{
  set<hwloc_obj_t> G;

  while(U.size() > 0) {
    vector< pair<hwloc_obj_t, float> > pick_k;

    foreach(hwloc_obj_t l, U) {
      set<hwloc_obj_t> Gltmp = G;
      Gltmp.insert(l);
      map<hwloc_obj_t, float>::iterator cl = cost.find(l);
      pick_k.push_back(
          make_pair(l, (f(info, Gltmp) - f(info, G)) / pow(cl->second, r))
      );
    }

    // find the cheapest core to add to the list
    //
    vector< pair<hwloc_obj_t, float> >::iterator k_itr =
        max_element(pick_k.begin(), pick_k.end(), pair_cmp);

    vector< pair<hwloc_obj_t, float> > cost_test;
    foreach(hwloc_obj_t obj_i, G) {
       map<hwloc_obj_t, float>::iterator ci = cost.find(obj_i);
       map<hwloc_obj_t, float>::iterator k_element = cost.find(k_itr->first);
       cost_test.push_back(
           make_pair(ci->first, ci->second + k_element->second)
       );
    }

    set<hwloc_obj_t> Gktmp = G;
    Gktmp.insert(k_itr->first);
    if( (accumulate(cost_test.begin(), cost_test.end(), 0.0, sum_func) <= B) &&
        ((f(info, Gktmp) - f(info, G)) >= 0.0) ) {
      G.insert(k_itr->first);
    }

    U.erase(k_itr->first);
  }

  vector< pair<hwloc_obj_t, float> > vlist;
  foreach(hwloc_obj_t v, V) {
     map<hwloc_obj_t, float>::iterator vitr = cost.find(v);
     set<hwloc_obj_t> vset;
     vset.insert(vitr->first);
     if(vitr->second <= B) {
       vlist.push_back( make_pair(vitr->first, f(info, vset)) );
     }
  }

  vector< pair<hwloc_obj_t, float> >::iterator vstar =
      max_element(vlist.begin(), vlist.end(), pair_cmp);

  set<hwloc_obj_t> vstarset;
  vstarset.insert(vstar->first);

  vector< pair<set<hwloc_obj_t>, float> > fin;
  fin.push_back( make_pair(vstarset, f(info, vstarset)) );
  fin.push_back( make_pair(G, f(info, G)) );

  vector< pair<set<hwloc_obj_t>, float> >::iterator finG =
      max_element(fin.begin(), fin.end(), fin_cmp);

  Gf = finG->first;
}

} // namespace slave {
} // namespace internal {
} // namespace mesos {
