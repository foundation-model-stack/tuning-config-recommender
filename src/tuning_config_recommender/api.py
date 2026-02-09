import asyncio
import os
import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import BackgroundTasks, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from tuning_config_recommender.adapters import FMSAdapter

app = FastAPI(title="Recommender API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def delete_files(file_paths: list[str]) -> None:
    await asyncio.sleep(600)
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")


class RecommendationsRequest(BaseModel):
    tuning_config: dict | None = None
    tuning_data_config: dict | None = None
    compute_config: dict | None = None
    accelerate_config: dict | None = None
    skip_estimator: bool | None = False


def generate_unique_stamps():
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    random_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_id}"


@app.post("/recommend")
async def recommend(
    background_tasks: BackgroundTasks,
    req: RecommendationsRequest,
):
    try:
        # Validate required fields
        if not req.tuning_config:
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder({"message": "tuning_config is required"}),
            )

        # Validate model name or path
        if not req.tuning_config.get("model_name_or_path"):
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder(
                    {"message": "model_name_or_path is required in tuning_config"}
                ),
            )

        # Validate training data path or data config
        if not req.tuning_data_config and not req.tuning_config.get(
            "training_data_path"
        ):
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder(
                    {
                        "message": "Either tuning_data_config or training_data_path in tuning_config is required"
                    }
                ),
            )

        paths_to_delete = []
        base_dir = Path(__file__).parent
        output_dir = base_dir / "outputs" / generate_unique_stamps()

        fms_adapter = FMSAdapter(base_dir=output_dir, additional_actions=[])

        response = fms_adapter.execute(
            tuning_config=req.tuning_config,
            compute_config=req.compute_config,
            accelerate_config=req.accelerate_config,
            data_config=req.tuning_data_config,
            unique_tag="",
            paths={},
            skip_estimator=req.skip_estimator,
        )
        response.pop("patches")
        for _, path in response["paths"].items():
            paths_to_delete.append(path)

        background_tasks.add_task(delete_files, paths_to_delete)
        return response
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"message": str(e)}),
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"message": str(e)}),
        )
    except OSError as e:
        logger.error(f"OSError: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"message": str(e)}),
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                {
                    "message": "An unexpected error occurred. Please try again or contact support."
                }
            ),
        )
