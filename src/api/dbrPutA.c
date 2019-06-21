/*
 * Copyright © 2018, 2019 IBM Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include "logutil.h"
#include "util/lock_tools.h"
#include "libdatabroker.h"
#include "libdatabroker_int.h"

#include <stdio.h>
#include <stdlib.h>

DBR_Tag_t libdbrPutA (DBR_Handle_t cs_handle,
                      dbrDA_Request_chain_t *request,
                      DBR_Group_t group)
{
  dbrName_space_t *cs = (dbrName_space_t*)cs_handle;
  if(( cs == NULL ) || ( cs->_be_ctx == NULL ) || ( cs->_reverse == NULL ) || (cs->_status != dbrNS_STATUS_REFERENCED ))
  {
    LOG( DBG_ERR, stderr, "Invalid input name space handle\n" );
    return DB_TAG_ERROR;
  }

  dbrDA_Request_chain_t *chain = request;

  BIGLOCK_LOCK( cs->_reverse );

  DBR_Tag_t tag = dbrTag_get( cs->_reverse );
  if( tag == DB_TAG_ERROR )
    BIGLOCK_UNLOCKRETURN( cs->_reverse, DB_TAG_ERROR );

#ifdef DBR_DATA_ADAPTERS
  // write-path data pre-processing plugin
  if( cs->_reverse->_data_adapter != NULL )
  {
    chain = cs->_reverse->_data_adapter->pre_write( request );
    if( chain == NULL )
      BIGLOCK_UNLOCKRETURN( cs->_reverse, DB_TAG_ERROR );
  }
#endif

  dbrRequestContext_t *head =
      dbrCreate_request_chain( DBBE_OPCODE_PUT,
                               cs_handle,
                               group,
                               NULL,
                               DBR_GROUP_EMPTY,
                               chain,
                               NULL,
                               0,
                               tag );
  if( head == NULL )
    return DB_TAG_ERROR;

  head->_rchain = chain; // potentially modified chain after plugin
  head->_ochain = request; // actual request chain from user
  DBR_Tag_t ptag = dbrInsert_request( cs, head );
  if( ptag == DB_TAG_ERROR )
  {
    LOG( DBG_ERR, stderr, "Unable to inject request to queue\n" );
    goto error;
  }

  DBR_Request_handle_t put_handle = dbrPost_request( head );
  if( put_handle == NULL )
    goto error;

  BIGLOCK_UNLOCKRETURN( cs->_reverse, head->_tag );

error:
  dbrRemove_request( cs, head );
#ifdef DBR_DATA_ADAPTERS
  if( cs->_reverse->_data_adapter != NULL )
    cs->_reverse->_data_adapter->error_handler( chain, DBRDA_WRITE, DBR_ERR_TAGERROR );
#endif
  BIGLOCK_UNLOCKRETURN( cs->_reverse, DB_TAG_ERROR );
}

