 #
 # Copyright (C) 2018 IBM Corporation
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #    http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 #
import os, sys
import dbr_module
from _dbr_interface import ffi
from dbr_module import dbr

dbr_name = "DBRtestname"
level = dbr.DBR_PERST_VOLATILE_SIMPLE
group_list = ffi.new('DBR_GroupList_t')
dbr_hdl = ffi.new('DBR_Handle_t*')
dbr_hdl = dbr.dbrCreate(dbr_name, level, group_list)


test_in = "Hello World!"
group = '0'
res = dbr.dbrPut(dbr_hdl, test_in, len(test_in), "testTup", group)
print 'Put ' + test_in
out_size = ffi.new('int64_t*')
out_size[0] = 1024

q = dbr.createBuf('char[]', out_size[0]) 
res = dbr.dbrRead(dbr_hdl, q, out_size, "testTup", "", group, dbr.DBR_FLAGS_NONE)
print 'Read returned: ' +  q[:]

res = dbr.dbrGet(dbr_hdl, q, out_size, "testTup", "", group, dbr.DBR_FLAGS_NONE)
print 'Get returned: ' + q[:]

print 'Delete Data Broker'
res = dbr.dbrDelete(dbr_name)
print 'Exit Status: ' + dbr.getErrorMessage(res)
