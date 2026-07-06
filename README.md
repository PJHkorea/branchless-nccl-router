# 🌌 branchless-nccl-router (v1.0)


> **"분산 통신망의 규칙을 다른 방식으로 생각해 봤습니다. 제어 흐름(Control Flow) 분기를 없애고, 대규모 클러스터의 집합 통신과 패킷 정화를 단일 융합 기계어 수식 레일로 해결하려 합니다."**

본 레포지토리는 다중 노드 분산 가속기 환경(Multi-Node Distributed Infrastructure)의 입구 경계면(Ingress Gate)에서 발생하는 제어 분기문(`if/else`)을 제거하고, **네트워크 동기화 지터(Synchronous Jitter) 비용을 물리적으로 0ns로 수축**시키기 위해 설계된 하드웨어 밀착형 분산 라우터 커널 모듈입니다.

---

## 🚀 3대 핵심 아키텍처 역학 (Core Architectural Mechanics)

### 1. jax.lax.psum과 제어문(if)의 동거 거부 (Zero-Jitter Collective Op)

- 다중 노드 분산 학습/추론 환경에서 엔지니어들을 가장 괴롭히는 난제는 "특정 노드가 패킷을 다 받았는지 확인하기 위해 멈춰 서는 하드웨어 동기화 펜스(Fence Stall) 병목"입니다. 
본 커널은 네트워크 패킷 유실 및 오염 여부를 판정하는 `if (is_packet_corrupted)` 같은 제어문을 제거했습니다. 대신 JAX/XLA 백엔드에 상주하는 NCCL All-Reduce 코어 프리미티브인 `jax.lax.psum` 연산 결과를 불린이 아닌 단정밀도 부동소수점 마스크 수식 `(global_sync_mask == 0.0).astype(target_dtype)` 내부의 피연산자로 직결했습니다. 32개 디바이스는 네트워크 전송 상태나 지터 발생 여부와 무관하게, 물리 NCCL 링(Ring) 레일 위를 동일한 속도로 움직이게 됩니다.

```python
# [The Core Magic] 제어 분기문을 제거하고 대수적 수축소멸 회로로 직결
global_sync_mask = jax.lax.psum(is_packet_corrupted, axis_name=cluster_axis_name)
is_mesh_clean = (global_sync_mask == literal_zero).astype(target_dtype)
next_jitter_flag = network_jitter_flag * is_mesh_clean
```

### 2. 다중 노드 SRAM Spill Over 방어형 클로저 스캔 (SRAM Register Locking)

- 32개 디바이스의 대규모 패킷 스트림 데이터를 루프 단계마다 통째로 들고 이동(Carry)하면 가속기는 고속 온칩 메모리 용량 부족으로 데이터를 글로벌 VRAM으로 빼두는 레지스터 스필오버(Register Spill) 병목이 생깁니다. 
본 엔진은 입력 스트림 행렬을 중첩 함수의 클로저(Closure) 영역에 박아두고 컴파일러 단에 정적 상수로 락을 걸었습니다. 그리고 오직 `jnp.take(..., dev_idx, axis=0)`라는 고속 가상 포인터 인덱싱 주소 변환만으로 데이터를 수집하도록 강제하여, XLA 컴파일러가 32개 분산 영토의 Stride Map 뼈대를 SRAM 레지스터 맵에 정적으로 고정 캐싱하도록 유도했습니다. 가변 패킷 유입 속에서도 메모리 버스 대역폭 낭비를 제거하는 것이 목표입니다.

### 3. 입구 단에서의 '통신-검증-라우팅' 삼위일체 통합 및 미분 절연

- 데이터 스트림이 모델 내부의 본진 레이어(의미론적 뇌)로 들어가기 직전, 이 최전방 인그레스 게이트웨이가  수식으로 1) 32개 디바이스 분산 배칭 셰이핑, 2) 하드웨어 NCCL 링 동기화 검증, 3) 오염 데이터 원천 증발 회로(`0.0 Matrix` 수축소멸)를 단일 융합 통신 명령어(Fused Collective Op) 파이프라인으로 처리합니다. 연산이 완료된 뒤 루프 바깥에서 단 한 번만 최소값 글로벌 리덕션(`jnp.min`)을 단행하여 전산망 대역폭을 최대한 사수하며, 최종 텔레메트리 지표를 `jax.lax.stop_gradient`로 완전히 밀봉하여 네트워크 노이즈가 모델의 백프로퍼게이션 미분 체인(Autograd Rule)을 오염시키는 역학을 절연하는 것이 목적입니다.
---


## 📜 License

```text
Copyright (c) 2026 PJHkorea. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
```

- **[EN]** This repository contains open-source code distributed under the **Apache License 2.0**. The full license text can be found in the `LICENSE` file.
- **[KR]** 본 저장소는 **Apache License 2.0으로 배포하는 소스코드**입니다. 전문은 루트 디렉토리의 `LICENSE` 파일에서 확인하실 수 있습니다.

---

> ⚠️ **[EN] Disclaimer**: All code within this repository is provided "AS IS", without warranty of any kind, express or implied.
>
> ⚠️ **[KR] 면책 조항**: 본 저장소의 모든 소스코드는 "있는 그대로(AS IS)" 제공되며, 명시적 또는 묵시적인 어떠한 보증도 제공하지 않습니다.
