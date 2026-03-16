# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.orm import Session

from airflow.models.dagbag import DBDagBag


class TestDBDagBagCacheEviction:
    """Tests for DBDagBag LRU cache eviction."""

    @staticmethod
    def _make_serdag(dag_version_id=None):
        """Create a mock SerializedDagModel with a fake dag."""
        serdag = MagicMock(spec_set=["dag", "dag_version_id", "load_op_links"])
        serdag.dag_version_id = dag_version_id or uuid4()
        serdag.dag = MagicMock()
        return serdag

    def test_unbounded_cache_default(self):
        """Default (max_cache_size=0) does not evict."""
        bag = DBDagBag(max_cache_size=0)
        for _ in range(100):
            bag._read_dag(self._make_serdag())
        assert len(bag._dags) == 100

    def test_cache_evicts_oldest_when_full(self):
        """With max_cache_size=3, adding a 4th entry evicts the oldest."""
        bag = DBDagBag(max_cache_size=3)

        ids = [uuid4() for _ in range(4)]
        for vid in ids:
            bag._read_dag(self._make_serdag(dag_version_id=vid))

        assert len(bag._dags) == 3
        # The first entry should have been evicted
        assert ids[0] not in bag._dags
        # The last 3 should be present
        assert ids[1] in bag._dags
        assert ids[2] in bag._dags
        assert ids[3] in bag._dags

    def test_cache_hit_refreshes_lru_order(self):
        """Accessing a cached entry moves it to the end, preventing eviction."""
        bag = DBDagBag(max_cache_size=3)

        ids = [uuid4() for _ in range(3)]
        for vid in ids:
            bag._read_dag(self._make_serdag(dag_version_id=vid))

        # Access ids[0] so it becomes most recently used
        result = bag._dags.get(ids[0])
        assert result is not None
        bag._dags.move_to_end(ids[0])

        # Add a 4th entry: should evict ids[1] (now the LRU), not ids[0]
        new_id = uuid4()
        bag._read_dag(self._make_serdag(dag_version_id=new_id))

        assert len(bag._dags) == 3
        assert ids[0] in bag._dags  # refreshed, not evicted
        assert ids[1] not in bag._dags  # LRU, evicted
        assert ids[2] in bag._dags
        assert new_id in bag._dags

    def test_get_dag_marks_as_recently_used(self):
        """_get_dag cache hit moves the entry to the end of the LRU."""
        bag = DBDagBag(max_cache_size=3)

        ids = [uuid4() for _ in range(3)]
        for vid in ids:
            bag._read_dag(self._make_serdag(dag_version_id=vid))

        # _get_dag on ids[0] should mark it as recently used
        session = MagicMock(spec=Session)
        result = bag._get_dag(ids[0], session=session)
        assert result is not None

        # Now add 2 more entries: should evict ids[1] and ids[2], not ids[0]
        bag._read_dag(self._make_serdag())
        bag._read_dag(self._make_serdag())

        assert len(bag._dags) == 3
        assert ids[0] in bag._dags

    def test_duplicate_key_updates_in_place(self):
        """Re-reading a DAG with the same version_id updates the entry without growing the cache."""
        bag = DBDagBag(max_cache_size=3)

        vid = uuid4()
        bag._read_dag(self._make_serdag(dag_version_id=vid))
        bag._read_dag(self._make_serdag(dag_version_id=vid))
        bag._read_dag(self._make_serdag(dag_version_id=vid))

        assert len(bag._dags) == 1

    def test_eviction_does_not_affect_none_dag(self):
        """_read_dag with a None dag does not add to cache."""
        bag = DBDagBag(max_cache_size=2)

        serdag = MagicMock(spec_set=["dag", "dag_version_id", "load_op_links"])
        serdag.dag_version_id = uuid4()
        serdag.dag = None

        result = bag._read_dag(serdag)
        assert result is None
        assert len(bag._dags) == 0
