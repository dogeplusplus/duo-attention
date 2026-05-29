.DEFAULT_GOAL := help

comma := ,

GIT_REF ?= $(shell git branch --show-current)
CONFIG_FILE ?= configs/hf_train.json
DOCKER ?= docker
DOCKER_IMAGE ?= kerorogunso/duo-attention-deps:cuda12.4
DOCKER_PLATFORM ?= linux/amd64
DOCKER_PREFETCH_MODEL ?= poolside/Laguna-XS.2
DOCKER_PREFETCH_REVISION ?=
DOCKER_BUILD_SECRETS ?= $(if $(HF_TOKEN),--secret id=hf_token$(comma)env=HF_TOKEN,)

BENCH_CONFIG_FILE ?= configs/laguna_benchmark.json
BENCH_JOB_CONFIG_FILE ?= configs/hf_laguna_benchmark_job.json
PUBLISH_REPO_ID ?= dogeplusplus/duo-laguna-adapter-smoke
PUBLISH_WANDB_PROJECT ?= dogeplusplus/DuoAttention
PUBLISH_ARTIFACT_DIR ?=
PUBLISH_PRIVATE ?= --private
PUBLISH_EXTRA ?=

.PHONY: help train update-docker-image docker-build docker-push benchmark-laguna-duo submit-laguna-benchmark publish-duo-adapter

help:
	@printf "Targets:\n"
	@printf "  make train                 Submit the HF Jobs smoke training run\n"
	@printf "  make update-docker-image   Build and push the Docker Hub dependency image\n"
	@printf "  make benchmark-laguna-duo  Benchmark base Laguna vs Duo Laguna locally\n"
	@printf "  make submit-laguna-benchmark Submit Laguna benchmark to HF Jobs\n"
	@printf "  make publish-duo-adapter   Publish latest W&B Duo adapter to the Hub\n"
	@printf "\nUseful overrides:\n"
	@printf "  CONFIG_FILE=... GIT_REF=... DOCKER_IMAGE=... BENCH_CONFIG_FILE=... PUBLISH_REPO_ID=...\n"

train:
	CONFIG_FILE="$(CONFIG_FILE)" \
	GIT_REF="$(GIT_REF)" \
	./scripts/run_training_hf.sh

update-docker-image: docker-build docker-push

docker-build:
	$(DOCKER) build \
		--platform "$(DOCKER_PLATFORM)" \
		--build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
		--build-arg PREFETCH_MODEL_ID="$(DOCKER_PREFETCH_MODEL)" \
		--build-arg PREFETCH_MODEL_REVISION="$(DOCKER_PREFETCH_REVISION)" \
		$(DOCKER_BUILD_SECRETS) \
		-t "$(DOCKER_IMAGE)" \
		-f Dockerfile .

docker-push:
	$(DOCKER) push "$(DOCKER_IMAGE)"

benchmark-laguna-duo:
	uv run python scripts/run_laguna_benchmark_config.py --config "$(BENCH_CONFIG_FILE)"

submit-laguna-benchmark:
	uv run python scripts/launch_hf_laguna_benchmark_job.py \
		--job-config "$(BENCH_JOB_CONFIG_FILE)" \
		--benchmark-config "$(BENCH_CONFIG_FILE)" \
		--git-ref "$(GIT_REF)"

publish-duo-adapter:
	uv run python scripts/publish_wandb_duo_adapter.py \
		--wandb-project "$(PUBLISH_WANDB_PROJECT)" \
		--repo-id "$(PUBLISH_REPO_ID)" \
		$(PUBLISH_PRIVATE) \
		$(if $(PUBLISH_ARTIFACT_DIR),--artifact-dir "$(PUBLISH_ARTIFACT_DIR)") \
		$(PUBLISH_EXTRA)
