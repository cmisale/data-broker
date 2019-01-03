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
import os
import sys

from _dbr_interface import ffi
from dbr_module import dbr


dbr_name = "DBRtestname"
level = dbr.DBR_PERST_VOLATILE_SIMPLE
group_list = ffi.new('DBR_GroupList_t')
dbr_hdl = ffi.new('DBR_Handle_t*')
dbr_hdl = dbr.dbrCreate(dbr_name, level, group_list)
group = '0'

# query the DBR to see if successful
dbr_state = ffi.new('DBR_State_t*')
res = dbr.dbrQuery(dbr_hdl, dbr_state, dbr.DBR_STATE_MASK_ALL)

test_in = "Hello World!"
res = dbr.dbrPut(dbr_hdl, test_in, len(test_in), "HelloTuple", group)
res = dbr.dbrPut(dbr_hdl, "Goodbye World!", len("Goodbye World!"), "GoodbyeTuple", group)

######
#test the directory command and list all tuple names/keys
out_size = ffi.new('int64_t*')
size = 1024
rsize = ffi.new('int64_t*')
count = 1000
res,keys = dbr.dbrDirectory(dbr_hdl, "*", group, count, size, rsize)

print 'Keys on DBR: ' + str(keys[:])

out_size[0] = 1024
q = dbr.createBuf('char[]', out_size[0])
group_t = 0
res = dbr.dbrRead(dbr_hdl, q, out_size, keys[0], "", group, dbr.DBR_FLAGS_NOWAIT)
print 'Read returned: ' +  q[:]
res = dbr.dbrGet(dbr_hdl, q, out_size, keys[0], "", group, dbr.DBR_FLAGS_NONE)
print 'Get returned: ' + q[:]
#read again to check for failing
out_size[0] = 1024
q = dbr.createBuf('char[]', out_size[0])
res = dbr.dbrRead(dbr_hdl, q, out_size, keys[1], "", group, dbr.DBR_FLAGS_NONE)
print 'Read returned: ' +  q[:]
res = dbr.dbrGet(dbr_hdl, q, out_size, keys[1], "", group, dbr.DBR_FLAGS_NONE)
print 'Get returned: ' + q[:]
print 'Delete Data Broker'
res = dbr.dbrDelete(dbr_name)
print 'Exit Status: ' + dbr.getErrorMessage(res)
