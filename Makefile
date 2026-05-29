.DEFAULT_GOAL := help

GIT_REF ?= $(shell git branch --show-current)
CONFIG_FILE ?= configs/hf_train.json
DOCKER ?= docker
DOCKER_IMAGE ?= kerorogunso/duo-attention-deps:cuda12.4
DOCKER_PLATFORM ?= linux/amd64

BENCH_MODEL ?= poolside/Laguna-XS.2
BENCH_FULL_ATTENTION_HEADS ?=
BENCH_OUTPUT ?= outputs/laguna_duo_benchmark.csv
BENCH_JSON_OUTPUT ?= outputs/laguna_duo_benchmark.jsonl
BENCH_PROMPT_LENGTHS ?= 128,512,1024,2048
BENCH_DECODE_LENGTHS ?= 1,16,64
BENCH_BATCH_SIZE ?= 1
BENCH_WARMUP ?= 1
BENCH_STEPS ?= 3
BENCH_DEVICE ?= cuda
BENCH_DTYPE ?= bfloat16
BENCH_ATTN_IMPLEMENTATION ?= eager
BENCH_SINK_SIZE ?= 128
BENCH_RECENT_SIZE ?= 256
BENCH_VARIANTS ?= base,duo
BENCH_EXTRA ?=

.PHONY: help train update-docker-image docker-build docker-push benchmark-laguna-duo

help:
	@printf "Targets:\n"
	@printf "  make train                 Submit the HF Jobs smoke training run\n"
	@printf "  make update-docker-image   Build and push the Docker Hub dependency image\n"
	@printf "  make benchmark-laguna-duo  Benchmark base Laguna vs Duo Laguna locally\n"
	@printf "\nUseful overrides:\n"
	@printf "  CONFIG_FILE=... GIT_REF=... DOCKER_IMAGE=... BENCH_FULL_ATTENTION_HEADS=...\n"

train:
	CONFIG_FILE="$(CONFIG_FILE)" \
	GIT_REF="$(GIT_REF)" \
	./scripts/run_training_hf.sh

update-docker-image: docker-build docker-push

docker-build:
	$(DOCKER) build \
		--platform "$(DOCKER_PLATFORM)" \
		--build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
		-t "$(DOCKER_IMAGE)" \
		-f Dockerfile .

docker-push:
	$(DOCKER) push "$(DOCKER_IMAGE)"

benchmark-laguna-duo:
	@if [ -z "$(BENCH_FULL_ATTENTION_HEADS)" ]; then \
		echo "Set BENCH_FULL_ATTENTION_HEADS=/path/to/full_attention_heads_latest.tsv"; \
		exit 1; \
	fi
	uv run python scripts/benchmark_laguna_duo.py \
		--model "$(BENCH_MODEL)" \
		--full-attention-heads "$(BENCH_FULL_ATTENTION_HEADS)" \
		--sink-size "$(BENCH_SINK_SIZE)" \
		--recent-size "$(BENCH_RECENT_SIZE)" \
		--prompt-lengths "$(BENCH_PROMPT_LENGTHS)" \
		--decode-lengths "$(BENCH_DECODE_LENGTHS)" \
		--batch-size "$(BENCH_BATCH_SIZE)" \
		--warmup "$(BENCH_WARMUP)" \
		--steps "$(BENCH_STEPS)" \
		--device "$(BENCH_DEVICE)" \
		--dtype "$(BENCH_DTYPE)" \
		--attn-implementation "$(BENCH_ATTN_IMPLEMENTATION)" \
		--variants "$(BENCH_VARIANTS)" \
		--output "$(BENCH_OUTPUT)" \
		--json-output "$(BENCH_JSON_OUTPUT)" \
		$(BENCH_EXTRA)
