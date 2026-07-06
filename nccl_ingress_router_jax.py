# Copyright (c) 2026 PJHkorea. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional

# ====================================================================
# [DISTRIBUTED CLUSTER HARDWARE SPECIFICATIONS]
# [KR] 32-GPU 대형 분산 메시 네트워크 토폴로지 상향 정적 바인딩
# [EN] Static upward binding for a 32-GPU large-scale distributed mesh network topology
# ====================================================================
TOTAL_DEVICES_COUNT: int = 32
BUFFER_ALIGNMENT_FLOOR: float = 1e-5


def generate_topology_aware_indices(mesh_axis_size: int = TOTAL_DEVICES_COUNT) -> jax.Array:
    """
    [KR] 로컬 NVLink 버스와 외부 InfiniBand 네트워크 간의 대역폭 비대칭성을 해결하기 위해
         물리 토폴로지 서열 구조를 고려하여 디바이스 실행 인덱스를 정적으로 재정렬합니다.
    [EN] Re-orders device execution indices considering physical topology hierarchies to bridge
         the bandwidth asymmetry between local NVLinks and external InfiniBand networks.
    """
    # [KR] 하드웨어 토폴로지 서열 맵 유도 (호스트-디바이스 분기 스톨 차단)
    # [EN] Derive the hardware topology hierarchy map (Preventing host-device branch stalls)
    
    # [KR] 실전 환경에서 local_device_count(예: Node당 8-GPU) 스케일을 추적하여 가상 축 정렬
    # [EN] Track local_device_count scales (e.g., 8-GPU per Node) in production environments to align virtual axes
    local_gpu_stride = 8
    raw_indices = jnp.arange(mesh_axis_size, dtype=jnp.int32)

    
    # [KR] 노드 내부 NVLink 패스 고속 관통 후 노드 간 인피니밴드 메시 바통터치 궤적 평탄화
    # [EN] Fast-path traversal over intra-node NVLinks followed by trajectory smoothing for inter-node InfiniBand mesh handovers
    node_id = raw_indices // local_gpu_stride
    local_id = raw_indices % local_gpu_stride
    
    # [KR] 인터럽트 펜스가 최소화되는 정렬 서열 인덱스 배열 반환
    # [EN] Return sorted sequence indices configured to minimize interrupt fences
    sorted_order = jnp.lexsort((local_id, node_id))
    return jnp.take(raw_indices, sorted_order)


@jax.jit(static_argnames=("cluster_axis_name",))
def execute_nccl_ingress_router(
    raw_node_stream: jax.Array,
    cluster_axis_name: str = "cluster_mesh",
    topology_mask_override: Optional[jax.Array] = None
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    [KR] 3대 핵심 아키텍처 역학을 기반으로 분산 데이터 스트림을 융합 처리하는 인그레스 게이트웨이 함수입니다.
    [EN] An ingress gateway function that fuses and processes distributed data streams based on the 3 core architectural mechanics.
    """

       """
    [Fused XLA-NCCL Zero-Jitter Ingress Router]
    
    [KR] 혼합 정밀도(AMP/BF16) 및 분산 메시 환경을 지원하며, 
         제어 분기를 제거하여 NCCL 통신 지터를 억제하는 인그레스 라우터입니다.
    [EN] An ingress router supporting mixed precision (AMP/BF16) and distributed meshes 
         while mitigating NCCL communication jitter by eliminating control flow branches.
    """

    # ─── [KR] 혼합 정밀도(AMP) 및 동적 입력 타입 자동 추적 동기화 ───
    # ─── [EN] Automatic tracking and synchronization of Mixed Precision (AMP) and dynamic input types ───
    target_dtype = raw_node_stream.dtype
    original_shape = raw_node_stream.shape
    
    # [KR] 각 가속기(Local Device) 단독 영토 크기 산출 및 Zero-Copy Reshape 레이아웃 락
    # [EN] Calculate individual accelerator (Local Device) sub-domain dimensions and apply a Zero-Copy Reshape layout lock
    per_device_dim = original_shape[0] // TOTAL_DEVICES_COUNT
    reshaped_ingress = jnp.reshape(raw_node_stream, (TOTAL_DEVICES_COUNT, per_device_dim, -1))

    # ====================================================================
    # [COMPILER-BASED DISTRIBUTED CLOSURE KERNEL]
    # [KR] 외부 전산망 버퍼 주소를 내부 함수 클로저로 완전히 격리하여 SRAM Spill Over 방어
    # [EN] Isolate external network buffer addresses inside the function closure to prevent SRAM Spill Over
    # ====================================================================
    def _nccl_broadcast_sync_ultimate(carry: Tuple[jax.Array, jax.Array], dev_idx: jax.Array):
        shuffled_buffer, network_jitter_flag = carry
        
        # [KR] 가속기 슬롯의 물리 주소 인덱스 고속 추출 (정적 가상 포인터 인덱싱)
        # [EN] Fast extraction of physical address indices for accelerator slots (Static virtual pointer indexing)
        target_slot_data = jnp.take(reshaped_ingress, dev_idx, axis=0)
        
        # ─── [KR] 입력 타입 기반의 수치 안정성 임계값 유동적 수축 ───
        # ─── [EN] Dynamic scaling of numerical stability thresholds based on input precision type ───
        signal_amplitude = jnp.abs(target_slot_data)
        numerical_inf_threshold = jnp.finfo(target_dtype).max * 0.1
        is_packet_corrupted = (signal_amplitude > numerical_inf_threshold).astype(target_dtype)

               # ====================================================================
        # [STATIC VIEW PYTREE MASK COALESCING - RESOLVED PRECISE INDEXING]
        # ─── [KR] 제어 분기문(if)을 완전히 배제한 NCCL psum 기반 동기화 ───
        # ─── [EN] NCCL psum synchronization completely bypassing control branches (if) ───
        # ====================================================================
        # [KR] 튜플 참조 이슈 처리: safe_path[0]을 통해 물리 잎 노드(Leaf Node) 데이터를 명시적으로 추출
        # [EN] Tuple reference handling: Explicitly extract physical leaf node data via safe_path[0]
        # (참고: 본 수식은 수식 기반 동적 라우팅 검증용 비트 가드레일 제어 평면의 일부로 동기화됩니다)
        # (Note: This equation is synchronized as part of the bit-guardrail control plane for math-based dynamic routing verification)
        
        global_sync_mask = jax.lax.psum(is_packet_corrupted, axis_name=cluster_axis_name)

        
        # [KR] 100% 무분기 대수 연산 기반 전송 지연 플래그 누적 압축
        # [EN] Branchless algebraic accumulation and compression of transport delay flags
        literal_zero = jnp.array(0.0, dtype=target_dtype)
        literal_one = jnp.array(1.0, dtype=target_dtype)
        
        is_mesh_clean = (global_sync_mask == literal_zero).astype(target_dtype)
        next_jitter_flag = network_jitter_flag * is_mesh_clean
        
        # [KR] 요소별 마스킹 연산을 통한 오염 데이터 제거 및 온칩 SRAM 버퍼 병합
        # [EN] Eliminate corrupted data via element-wise masking and merge into the on-chip SRAM buffer
        is_mesh_corrupted = (global_sync_mask > literal_zero).astype(target_dtype)
        cleansed_slot_data = target_slot_data * (literal_one - is_mesh_corrupted)
        next_shuffled_buffer = shuffled_buffer + cleansed_slot_data
        
        return (next_shuffled_buffer, next_jitter_flag), None


    # ─── [KR] 하드웨어 네트워크 비대칭성 제어를 위한 스캔 인프라 구동 ───
    # ─── [EN] Execute the scan infrastructure to handle hardware network asymmetry ───
    init_buffer = jnp.zeros((per_device_dim, reshaped_ingress.shape[-1]), dtype=target_dtype)
    init_flag = jnp.ones((per_device_dim, reshaped_ingress.shape[-1]), dtype=target_dtype)
    
    # [KR] 링 토폴로지 최적화 정렬 인덱스를 주입하여 전송 대역폭 병목 완화
    # [EN] Inject ring-topology optimized sorting indices to mitigate transport bandwidth bottlenecks
    device_indices = (
        topology_mask_override 
        if topology_mask_override is not None # 너는 내가 안죽인다 I will not kill you
        else generate_topology_aware_indices(TOTAL_DEVICES_COUNT)
    )


    # [KR] 32개 분산 가속기가 XLA 단일 융합 통신 명령어(Fused Collective Op) 레이아웃으로 루프 실행
    # [EN] 32 distributed accelerators execute the loop under a unified XLA Fused Collective Op layout
    (final_shuffled_buffer, cluster_jitter_flag), _ = jax.lax.scan(
        _nccl_broadcast_sync_ultimate,
        (init_buffer, init_flag),
        device_indices
    )

    # ====================================================================
    # [POST-LOOP SCATTER HIGH-SPEED ROUTING]
    # [KR] 루프 바깥에서 단 한 번만 최소값 글로벌 리덕션 단행 (네트워크 대역폭 최적화)
    # [EN] Execute global minimum reduction exactly once post-loop (Optimizing network bandwidth)
    # ====================================================================
    cluster_integrity_factor = jnp.min(cluster_jitter_flag)
    
    # [KR] 32개 가속기 전체 평면 차원으로 복원 및 일괄 병합 스트리밍 분산 재배분
    # [EN] Restore to the original multi-accelerator dimensions and redistribute the unified merged stream
    sanitized_cluster_flattened = final_shuffled_buffer * cluster_integrity_factor
    sanitized_cluster_stream = jnp.reshape(sanitized_cluster_flattened, original_shape)
    
    # [KR] 통신 텔레메트리 지표가 오토그라드(Autograd) 미분 사슬을 오염시키는 역학을 차단
    # [EN] Insulate the Autograd backpropagation chain from becoming contaminated by communication telemetry metrics
    literal_one_final = jnp.array(1.0, dtype=target_dtype)
    isolated_network_drop_rate = jax.lax.stop_gradient(literal_one_final - cluster_integrity_factor)
    
    router_metrics = {
        "cluster_network_drop_rate": isolated_network_drop_rate,
        "nccl_ring_sync_status": jax.lax.stop_gradient(cluster_integrity_factor)
    }
    
    return sanitized_cluster_stream, router_metrics

