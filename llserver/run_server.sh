#!/bin/bash

conda activate llama2
cd llserver/server
uvicorn server:app --reload