.DEFAULT_GOAL := help

GIT_REPO_URL ?= https://github.com/dogeplusplus/duo-attention.git
GIT_REF ?= $(shell git branch --show-current)
HF_JOB_IMAGE ?= pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

MODEL_NAME ?= poolside/Laguna-XS.2
HF_FLAVOR ?= a100-large
TIMEOUT ?= 1h
NPROC_PER_NODE ?= 1

NUM_STEPS ?= 1
MAX_LENGTH ?= 768
CONTEXT_LENGTH_MIN ?= $(MAX_LENGTH)
CONTEXT_LENGTH_MAX ?= $(MAX_LENGTH)

.PHONY: help train

help:
	@printf "Targets:\n"
	@printf "  make train                 Submit the HF Jobs smoke training run\n"
	@printf "\nUseful overrides:\n"
	@printf "  GIT_REF=... MODEL_NAME=... HF_FLAVOR=... HF_JOB_IMAGE=...\n"

train:
	GIT_REPO_URL="$(GIT_REPO_URL)" \
	GIT_REF="$(GIT_REF)" \
	HF_JOB_IMAGE="$(HF_JOB_IMAGE)" \
	PREINSTALLED_DEPS=0 \
	MODEL_NAME="$(MODEL_NAME)" \
	HF_FLAVOR="$(HF_FLAVOR)" \
	TIMEOUT="$(TIMEOUT)" \
	NPROC_PER_NODE="$(NPROC_PER_NODE)" \
	NUM_STEPS="$(NUM_STEPS)" \
	MAX_LENGTH="$(MAX_LENGTH)" \
	CONTEXT_LENGTH_MIN="$(CONTEXT_LENGTH_MIN)" \
	CONTEXT_LENGTH_MAX="$(CONTEXT_LENGTH_MAX)" \
	./scripts/run_training_hf.sh
