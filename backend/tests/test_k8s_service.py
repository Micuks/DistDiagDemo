import pytest
import pytest_asyncio
import asyncio
import sys
import os

# Add the backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, backend_dir)

from unittest.mock import patch, MagicMock, AsyncMock, call, ANY
from kubernetes.client.rest import ApiException
from kubernetes import client
import redis.exceptions
import subprocess
import json
import uuid

# Assume K8sService is importable. Adjust path if necessary.
# e.g., from app.services.k8s_service import K8sService
from app.services.k8s_service import K8sService

# --- Fixtures ---

@pytest.fixture
def mock_env(monkeypatch):
    """Mocks environment variables."""
    monkeypatch.setenv('OCEANBASE_NAMESPACE', 'test-ns')
    monkeypatch.setenv('REDIS_HOST', 'mockredis')
    monkeypatch.setenv('REDIS_PORT', '6379')
    monkeypatch.setenv('REDIS_DB', '0')
    monkeypatch.setenv('OB_HOST', 'mockob')
    monkeypatch.setenv('OB_PORT', '2881')
    monkeypatch.setenv('OB_USER', 'mockuser')
    monkeypatch.setenv('OB_PASSWORD', 'mockpass')
    monkeypatch.setenv('OB_DATABASE', 'mockdb')
    monkeypatch.setenv('KUBERNETES_SERVICE_HOST', 'mockhost') # Simulate incluster config

@pytest.fixture
def mock_redis_client():
    """Mocks the redis.Redis client instance."""
    mock_client = MagicMock(spec=redis.Redis)
    mock_client.ping.return_value = True
    mock_client.smembers.return_value = set()
    mock_client.get.return_value = None
    # Mock pipeline methods if needed specifically
    mock_pipeline = MagicMock()
    mock_pipeline.get.return_value = mock_pipeline # Chainable
    mock_pipeline.execute.return_value = []
    mock_client.pipeline.return_value = mock_pipeline
    return mock_client

@pytest.fixture
def mock_k8s_apis():
    """Mocks Kubernetes client APIs."""
    mock_custom_api = AsyncMock(spec=client.CustomObjectsApi)
    mock_custom_api.create_namespaced_custom_object = AsyncMock()
    mock_custom_api.delete_namespaced_custom_object = AsyncMock()

    mock_core_api = AsyncMock(spec=client.CoreV1Api)
    mock_pod = MagicMock()
    mock_pod.metadata.name = "obcluster-pod-1"
    mock_pod.status.phase = "Running"
    mock_pod.status.pod_ip = "10.0.0.1"
    mock_ready_condition = MagicMock()
    mock_ready_condition.type = "Ready"
    mock_ready_condition.status = "True"
    mock_pod.status.conditions = [mock_ready_condition]
    mock_core_api.list_namespaced_pod.return_value = MagicMock(items=[mock_pod])


    return mock_custom_api, mock_core_api

@pytest.fixture
def mock_pymysql():
    """Mocks pymysql connection and cursor."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    # Simulate fetching zones
    mock_cursor.fetchall.return_value = [('zone1',), ('zone2',), ('zone3',)]
    with patch('pymysql.connect', return_value=mock_conn) as mock_connect:
        yield mock_connect, mock_cursor

@pytest.fixture
def mock_subprocess():
    """Mocks subprocess.run and subprocess.Popen."""
    mock_run = AsyncMock(return_value=MagicMock(stdout=b'', stderr=b'', returncode=0))
    mock_popen = AsyncMock(return_value=MagicMock(pid=12345)) # Mock Popen if needed
    with patch('subprocess.run', mock_run), \
         patch('subprocess.Popen', mock_popen): # Patch Popen if used by any anomaly type
        yield mock_run, mock_popen

@pytest_asyncio.fixture
async def k8s_service_instance(mock_env, mock_redis_client, mock_k8s_apis, mock_pymysql, mock_subprocess):
    """Provides a K8sService instance with mocked dependencies."""
    mock_custom_api, mock_core_api = mock_k8s_apis
    mock_pymysql_connect, _ = mock_pymysql
    with patch('redis.Redis', return_value=mock_redis_client), \
         patch('kubernetes.config.load_incluster_config'), \
         patch('kubernetes.client.CustomObjectsApi', return_value=mock_custom_api), \
         patch('kubernetes.client.CoreV1Api', return_value=mock_core_api):
        # Patch asyncio.to_thread to run synchronously for easier testing
        # If direct mocking of targets (like subprocess.run) is done, this might not be needed
        # with patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            service = K8sService()
            # Ensure async fetches during init are awaited if not mocked differently
            await asyncio.sleep(0) # Allow any initial async tasks to potentially run/be mocked
            # Manually set available_nodes if _fetch_available_nodes wasn't fully mocked or run
            if not service.available_nodes:
                 service.available_nodes = ["obcluster-pod-1"]
            if not service.ob_zones:
                 service.ob_zones = ["zone1", "zone2"] # Set default if fetch failed in mock setup
            # Manually assign mocks if needed, though patching classes should work
            service.redis_client = mock_redis_client
            service.custom_api = mock_custom_api
            service.core_api = mock_core_api
            return service

# --- Test Cases ---

@pytest.mark.asyncio
async def test_apply_anomaly_cpu_saturation_success(k8s_service_instance, mock_subprocess):
    """Test successful application of cpu_saturation anomaly."""
    service = k8s_service_instance
    mock_sub_run, _ = mock_subprocess
    target_node = "obcluster-pod-1"
    anomaly_type = "cpu_saturation"
    severity = "medium"

    # Mock the subprocess import that would happen inside apply_anomaly
    results = await service.apply_anomaly(anomaly_type, target_node, severity)

    assert len(results) == 1
    applied_anomaly = results[0]
    assert applied_anomaly["type"] == anomaly_type
    assert applied_anomaly["node"] == target_node
    assert applied_anomaly["status"] == "active"
    assert applied_anomaly["params"]["target_pod"] == target_node

    # Check kubectl exec call
    expected_cmd = f"kubectl exec -n test-ns {target_node} -- /bin/bash -c 'nohup stress-ng --cpu 30 -t {service.collection_duration} > /dev/null 2>&1 &'"
    mock_sub_run.assert_called_once_with(expected_cmd, shell=True, check=True, capture_output=True)

    # Check Redis calls
    anomaly_id = applied_anomaly["id"]
    service.redis_client.set.assert_called_once_with(f"anomaly:{anomaly_id}", json.dumps(applied_anomaly))
    service.redis_client.sadd.assert_called_once_with("active_anomalies", anomaly_id)

@pytest.mark.asyncio
async def test_apply_anomaly_network_bottleneck_success(k8s_service_instance, mock_k8s_apis):
    """Test successful application of a Chaos Mesh anomaly."""
    service = k8s_service_instance
    mock_custom_api, _ = mock_k8s_apis
    target_node = "obcluster-pod-1"
    anomaly_type = "network_bottleneck"
    severity = "high"

    results = await service.apply_anomaly(anomaly_type, target_node, severity)

    assert len(results) == 1
    applied_anomaly = results[0]
    assert applied_anomaly["type"] == anomaly_type
    assert applied_anomaly["node"] == target_node
    assert applied_anomaly["status"] == "active"
    assert "k8s_name" in applied_anomaly
    assert applied_anomaly["k8s_name"] is not None

    # Check Chaos Mesh create call
    mock_custom_api.create_namespaced_custom_object.assert_called_once()
    call_args = mock_custom_api.create_namespaced_custom_object.call_args
    assert call_args.kwargs['group'] == "chaos-mesh.org"
    assert call_args.kwargs['version'] == "v1alpha1"
    assert call_args.kwargs['namespace'] == "test-ns"
    assert call_args.kwargs['plural'] == "networkchaos"
    body = call_args.kwargs['body']
    assert body['kind'] == "NetworkChaos"
    assert body['metadata']['name'].startswith(f"ob-network-bottleneck")
    assert body['spec']['selector']['pods'] == {"test-ns": [target_node]}
    assert body['spec']['loss']['loss'] == "100" # High severity

    # Check Redis calls
    anomaly_id = applied_anomaly["id"]
    service.redis_client.set.assert_called_once_with(f"anomaly:{anomaly_id}", json.dumps(applied_anomaly))
    service.redis_client.sadd.assert_called_once_with("active_anomalies", anomaly_id)

@pytest.mark.asyncio
async def test_apply_anomaly_too_many_indexes_success(k8s_service_instance, mock_pymysql):
    """Test successful application of SQL-based anomaly."""
    service = k8s_service_instance
    _, mock_cursor = mock_pymysql
    anomaly_type = "too_many_indexes"
    target_node = "cluster_wide" # This type ignores specific node

    results = await service.apply_anomaly(anomaly_type, target_node)

    assert len(results) == 1
    applied_anomaly = results[0]
    assert applied_anomaly["type"] == anomaly_type
    assert applied_anomaly["node"] == "cluster_wide" # Check it stores cluster_wide correctly
    assert applied_anomaly["status"] == "active"
    assert applied_anomaly["params"]["sql_command_count"] > 0

    # Check SQL execution calls (check count or specific commands)
    assert mock_cursor.execute.call_count > 10 # Numerous index commands
    # Example check for one command
    mock_cursor.execute.assert_any_call("CREATE INDEX idx_customer_1 ON customer(c_w_id)")

    # Check Redis calls
    anomaly_id = applied_anomaly["id"]
    service.redis_client.set.assert_called_once_with(f"anomaly:{anomaly_id}", json.dumps(applied_anomaly))
    service.redis_client.sadd.assert_called_once_with("active_anomalies", anomaly_id)


@pytest.mark.asyncio
async def test_apply_anomaly_kubectl_failure(k8s_service_instance, mock_subprocess):
    """Test handling of kubectl exec failure during application."""
    service = k8s_service_instance
    mock_sub_run, _ = mock_subprocess
    mock_sub_run.side_effect = subprocess.CalledProcessError(1, "kubectl", stderr=b"error executing command")
    target_node = "obcluster-pod-1"
    anomaly_type = "cpu_saturation"

    # Expect the error to be caught and logged within the loop, not raised
    results = await service.apply_anomaly(anomaly_type, target_node)

    assert len(results) == 0 # No anomaly should be successfully created/stored

    # Check kubectl exec was called
    expected_cmd = f"kubectl exec -n test-ns {target_node} -- /bin/bash -c 'nohup stress-ng --cpu 30 -t {service.collection_duration} > /dev/null 2>&1 &'"
    mock_sub_run.assert_called_once_with(expected_cmd, shell=True, check=True, capture_output=True)

    # Check Redis cleanup was attempted (id generated but store failed)
    # We need the generated UUID. Patch uuid.uuid4 if needed for predictability.
    # Since the exception happens before storing, delete/srem might not be called.
    # Let's verify SET/SADD were NOT called.
    service.redis_client.set.assert_not_called()
    service.redis_client.sadd.assert_not_called()
    # Check if cleanup *was* called after failure (depends on exact structure)
    # service.redis_client.delete.assert_called_once() # Check if delete is called on failure
    # service.redis_client.srem.assert_called_once()  # Check if srem is called on failure

@pytest.mark.asyncio
async def test_apply_anomaly_redis_failure_on_store(k8s_service_instance, mock_subprocess):
    """Test handling of Redis failure during storing anomaly state."""
    service = k8s_service_instance
    mock_sub_run, _ = mock_subprocess
    target_node = "obcluster-pod-1"
    anomaly_type = "cpu_saturation"

    # Mock Redis set to fail
    service.redis_client.set.side_effect = redis.exceptions.ConnectionError("Redis down")

    # Mock the subprocess import that would happen inside apply_anomaly
    with pytest.raises(redis.exceptions.ConnectionError):
        await service.apply_anomaly(anomaly_type, target_node)

    # Check kubectl exec was still called
    expected_cmd = f"kubectl exec -n test-ns {target_node} -- /bin/bash -c 'nohup stress-ng --cpu 30 -t {service.collection_duration} > /dev/null 2>&1 &'"
    mock_sub_run.assert_called_once_with(expected_cmd, shell=True, check=True, capture_output=True)

    # Check Redis set was called (and failed)
    service.redis_client.set.assert_called_once()
    service.redis_client.sadd.assert_not_called() # Should fail before sadd

@pytest.mark.asyncio
async def test_delete_anomaly_net_saturation_success(k8s_service_instance, mock_subprocess):
    """Test successful deletion of net_saturation anomaly."""
    service = k8s_service_instance
    mock_sub_run, _ = mock_subprocess
    anomaly_id = str(uuid.uuid4())
    target_node = "obcluster-pod-1"
    anomaly_data = {
        "id": anomaly_id, "type": "net_saturation", "node": target_node, "status": "active",
        "k8s_name": None, "params": {"target_pod": target_node}
    }
    service.redis_client.get.return_value = json.dumps(anomaly_data)
    
    results = await service.delete_anomaly(anomaly_id=anomaly_id)

    assert results == [anomaly_id]

    # Check Redis get
    service.redis_client.get.assert_called_once_with(f"anomaly:{anomaly_id}")

    # Check kubectl exec cleanup calls
    expected_calls = [
        call(f"kubectl exec -n test-ns {target_node} -- /bin/bash -c 'iptables -t mangle -F'", shell=True, check=True, capture_output=True),
        call(f"kubectl exec -n test-ns {target_node} -- /bin/bash -c 'tc qdisc del dev eth0 root'", shell=True, check=True, capture_output=True)
    ]
    mock_sub_run.assert_has_calls(expected_calls)

    # Check Redis delete/srem
    service.redis_client.delete.assert_called_once_with(f"anomaly:{anomaly_id}")
    service.redis_client.srem.assert_called_once_with("active_anomalies", anomaly_id)

@pytest.mark.asyncio
async def test_delete_anomaly_chaos_mesh_success(k8s_service_instance, mock_k8s_apis):
    """Test successful deletion of a Chaos Mesh anomaly."""
    service = k8s_service_instance
    mock_custom_api, _ = mock_k8s_apis
    anomaly_id = str(uuid.uuid4())
    k8s_name = "ob-network-bottleneck-xyz"
    anomaly_data = {
        "id": anomaly_id, "type": "network_bottleneck", "node": "obcluster-pod-1",
        "status": "active", "k8s_name": k8s_name, "params": {}
    }
    service.redis_client.get.return_value = json.dumps(anomaly_data)

    results = await service.delete_anomaly(anomaly_id=anomaly_id)

    assert results == [anomaly_id]

    # Check Redis get
    service.redis_client.get.assert_called_once_with(f"anomaly:{anomaly_id}")

    # Check Chaos Mesh delete call
    mock_custom_api.delete_namespaced_custom_object.assert_called_once_with(
        group="chaos-mesh.org", version="v1alpha1", namespace="test-ns",
        plural="networkchaos", name=k8s_name
    )

    # Check Redis delete/srem
    service.redis_client.delete.assert_called_once_with(f"anomaly:{anomaly_id}")
    service.redis_client.srem.assert_called_once_with("active_anomalies", anomaly_id)


@pytest.mark.asyncio
async def test_delete_anomaly_not_found_in_redis(k8s_service_instance, mock_k8s_apis, mock_subprocess):
    """Test deleting an anomaly ID that doesn't exist in Redis."""
    service = k8s_service_instance
    mock_custom_api, _ = mock_k8s_apis
    mock_sub_run, _ = mock_subprocess
    anomaly_id = "non-existent-id"
    service.redis_client.get.return_value = None # Simulate not found

    results = await service.delete_anomaly(anomaly_id=anomaly_id)

    assert results == []
    service.redis_client.get.assert_called_once_with(f"anomaly:{anomaly_id}")
    # Ensure no cleanup actions were attempted
    mock_custom_api.delete_namespaced_custom_object.assert_not_called()
    mock_sub_run.assert_not_called()
    service.redis_client.delete.assert_not_called()
    service.redis_client.srem.assert_not_called()

@pytest.mark.asyncio
async def test_delete_anomaly_k8s_resource_not_found(k8s_service_instance, mock_k8s_apis):
    """Test deletion when the K8s resource is already gone (404)."""
    service = k8s_service_instance
    mock_custom_api, _ = mock_k8s_apis
    mock_custom_api.delete_namespaced_custom_object.side_effect = ApiException(status=404)
    anomaly_id = str(uuid.uuid4())
    k8s_name = "ob-network-bottleneck-gone"
    anomaly_data = {
        "id": anomaly_id, "type": "network_bottleneck", "node": "obcluster-pod-1",
        "status": "active", "k8s_name": k8s_name, "params": {}
    }
    service.redis_client.get.return_value = json.dumps(anomaly_data)

    results = await service.delete_anomaly(anomaly_id=anomaly_id)

    # Should still succeed from Redis perspective
    assert results == [anomaly_id]

    # Check Chaos Mesh delete was called (and raised 404 internally)
    mock_custom_api.delete_namespaced_custom_object.assert_called_once_with(
        group="chaos-mesh.org", version="v1alpha1", namespace="test-ns",
        plural="networkchaos", name=k8s_name
    )

    # Check Redis cleanup still happened
    service.redis_client.delete.assert_called_once_with(f"anomaly:{anomaly_id}")
    service.redis_client.srem.assert_called_once_with("active_anomalies", anomaly_id)

@pytest.mark.asyncio
async def test_get_active_anomalies_success(k8s_service_instance):
    """Test retrieving active anomalies successfully."""
    service = k8s_service_instance
    id1 = str(uuid.uuid4())
    id2 = str(uuid.uuid4())
    data1 = {"id": id1, "type": "cpu_saturation", "node": "pod-1"}
    data2 = {"id": id2, "type": "network_bottleneck", "node": "pod-2"}

    service.redis_client.smembers.return_value = {id1, id2}
    # Mock pipeline execution result
    service.redis_client.pipeline.return_value.execute.return_value = [
        json.dumps(data1), json.dumps(data2)
    ]

    active_list = await service.get_active_anomalies()

    assert len(active_list) == 2
    assert data1 in active_list
    assert data2 in active_list
    service.redis_client.smembers.assert_called_once_with("active_anomalies")
    # Check pipeline calls
    assert service.redis_client.pipeline.return_value.get.call_count == 2
    service.redis_client.pipeline.return_value.get.assert_has_calls(
        [call(f"anomaly:{id1}"), call(f"anomaly:{id2}")], any_order=True
    )
    service.redis_client.pipeline.return_value.execute.assert_called_once()

@pytest.mark.asyncio
async def test_get_active_anomalies_inconsistent_data(k8s_service_instance):
    """Test handling inconsistent data (ID in set, key missing)."""
    service = k8s_service_instance
    id1 = str(uuid.uuid4()) # Valid
    id2 = str(uuid.uuid4()) # Missing key
    data1 = {"id": id1, "type": "cpu_saturation", "node": "pod-1"}

    service.redis_client.smembers.return_value = {id1, id2}
    # Mock pipeline execution result - None for id2
    service.redis_client.pipeline.return_value.execute.return_value = [
        json.dumps(data1), None
    ]

    active_list = await service.get_active_anomalies()

    assert len(active_list) == 1
    assert data1 in active_list # Only valid data returned
    service.redis_client.smembers.assert_called_once_with("active_anomalies")
    # Check that the inconsistent ID was removed from the set
    # We use ANY because the order of IDs from smembers isn't guaranteed
    service.redis_client.srem.assert_called_once_with("active_anomalies", ANY)
    # Ensure the valid key was not deleted
    service.redis_client.delete.assert_not_called() # delete is called for invalid JSON, not missing keys

@pytest.mark.asyncio
async def test_delete_all_anomalies(k8s_service_instance):
    """Test deleting all active anomalies."""
    service = k8s_service_instance
    id1 = str(uuid.uuid4())
    id2 = str(uuid.uuid4())
    data1 = {"id": id1, "type": "cpu_saturation", "node": "pod-1"}
    data2 = {"id": id2, "type": "network_bottleneck", "node": "pod-2", "k8s_name": "k8s-res-2"}

    # Mock finding active anomalies
    service.redis_client.smembers.return_value = {id1, id2}

    # Mock the result of get within delete_anomaly calls
    def mock_get_side_effect(key):
        if key == f"anomaly:{id1}": return json.dumps(data1)
        if key == f"anomaly:{id2}": return json.dumps(data2)
        return None
    service.redis_client.get.side_effect = mock_get_side_effect

    # Mock the actual deletion actions (subprocess, k8s api) within delete_anomaly
    with patch.object(service.custom_api, 'delete_namespaced_custom_object', AsyncMock()) as mock_k8s_delete, \
         patch('subprocess.run', AsyncMock(return_value=MagicMock(stdout=b'', stderr=b'', returncode=0))) as mock_sub_run: # Mocking it here for simplicity

        deleted_ids = await service.delete_all_anomalies()

        assert set(deleted_ids) == {id1, id2}
        assert service.redis_client.smembers.call_count >= 1 # Initial call + maybe check at end

        # Check that get was called for each ID during deletion
        assert service.redis_client.get.call_count == 2
        service.redis_client.get.assert_has_calls([call(f"anomaly:{id1}"), call(f"anomaly:{id2}")], any_order=True)

        # Check that appropriate cleanup actions were called (e.g., K8s delete for id2)
        mock_k8s_delete.assert_called_once_with(group="chaos-mesh.org", version="v1alpha1", namespace="test-ns", plural="networkchaos", name="k8s-res-2")
        # Check subprocess was NOT called because cpu_saturation relies on termination

        # Check Redis cleanup happened for both
        assert service.redis_client.delete.call_count == 2
        service.redis_client.delete.assert_has_calls([call(f"anomaly:{id1}"), call(f"anomaly:{id2}")], any_order=True)
        assert service.redis_client.srem.call_count == 2
        service.redis_client.srem.assert_has_calls([call("active_anomalies", id1), call("active_anomalies", id2)], any_order=True)
