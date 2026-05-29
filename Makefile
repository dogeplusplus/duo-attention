.DEFAULT_GOAL := help

DOCKER ?= docker

IMAGE_REPO ?= ghcr.io/dogeplusplus/duo-attention-deps
IMAGE_TAG ?= cuda12.4
HF_JOB_IMAGE ?= $(IMAGE_REPO):$(IMAGE_TAG)

GIT_REPO_URL ?= https://github.com/dogeplusplus/duo-attention.git
GIT_REF ?= $(shell git branch --show-current)

MODEL_NAME ?= poolside/Laguna-XS.2
HF_FLAVOR ?= a100-large
TIMEOUT ?= 1h
NPROC_PER_NODE ?= 1

NUM_STEPS ?= 1
MAX_LENGTH ?= 768
CONTEXT_LENGTH_MIN ?= $(MAX_LENGTH)
CONTEXT_LENGTH_MAX ?= $(MAX_LENGTH)

.PHONY: help train update-docker-image docker-build docker-push

help:
	@printf "Targets:\n"
	@printf "  make train                 Submit the HF Jobs smoke training run\n"
	@printf "  make update-docker-image   Build and push the dependency-only Docker image\n"
	@printf "\nUseful overrides:\n"
	@printf "  IMAGE_REPO=... IMAGE_TAG=... GIT_REF=... MODEL_NAME=... HF_FLAVOR=...\n"

train:
	GIT_REPO_URL="$(GIT_REPO_URL)" \
	GIT_REF="$(GIT_REF)" \
	HF_JOB_IMAGE="$(HF_JOB_IMAGE)" \
	PREINSTALLED_DEPS=1 \
	MODEL_NAME="$(MODEL_NAME)" \
	HF_FLAVOR="$(HF_FLAVOR)" \
	TIMEOUT="$(TIMEOUT)" \
	NPROC_PER_NODE="$(NPROC_PER_NODE)" \
	NUM_STEPS="$(NUM_STEPS)" \
	MAX_LENGTH="$(MAX_LENGTH)" \
	CONTEXT_LENGTH_MIN="$(CONTEXT_LENGTH_MIN)" \
	CONTEXT_LENGTH_MAX="$(CONTEXT_LENGTH_MAX)" \
	./scripts/run_training_hf.sh

update-docker-image: docker-build docker-push

docker-build:
	$(DOCKER) build \
		--build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
		-t "$(HF_JOB_IMAGE)" \
		-f Dockerfile .

docker-push:
	$(DOCKER) push "$(HF_JOB_IMAGE)"
