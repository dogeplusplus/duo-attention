.DEFAULT_GOAL := help

GIT_REF ?= $(shell git branch --show-current)
CONFIG_FILE ?= configs/hf_train.json
DOCKER ?= docker
DOCKER_IMAGE ?= kerorogunso/duo-attention-deps:cuda12.4
DOCKER_PLATFORM ?= linux/amd64

BENCH_CONFIG_FILE ?= configs/laguna_benchmark.json

.PHONY: help train update-docker-image docker-build docker-push benchmark-laguna-duo

help:
	@printf "Targets:\n"
	@printf "  make train                 Submit the HF Jobs smoke training run\n"
	@printf "  make update-docker-image   Build and push the Docker Hub dependency image\n"
	@printf "  make benchmark-laguna-duo  Benchmark base Laguna vs Duo Laguna locally\n"
	@printf "\nUseful overrides:\n"
	@printf "  CONFIG_FILE=... GIT_REF=... DOCKER_IMAGE=... BENCH_CONFIG_FILE=...\n"

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
	uv run python scripts/run_laguna_benchmark_config.py --config "$(BENCH_CONFIG_FILE)"
