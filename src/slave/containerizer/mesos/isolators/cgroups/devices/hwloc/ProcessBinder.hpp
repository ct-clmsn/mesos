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

#ifndef __MESOSPROCESSBINDER__
#define __MESOSPROCESSBINDER__ 1

#include "stout/try.hpp"
#include "hwloc.hpp"

namespace mesos {
namespace internal {
namespace slave {


Try<bool> pin_executor_cpu(TopologyResourceInformation& loc,
                           const pid_t pid,
                           const string cgroup,
                           const float ncpus_req,
                           const float ngpus_req);

Try<bool> unpin_executor_cpu(TopologyResourceInformation& loc,
                           const pid_t pid);

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif
