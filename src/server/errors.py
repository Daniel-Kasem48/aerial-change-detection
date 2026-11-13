"""Domain errors raised by services, mapped to HTTP responses in one place."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI, Request


class ScannerError(Exception):
    status_code = 400

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ScanNotFoundError(ScannerError):
    status_code = 404


class ExampleNotFoundError(ScannerError):
    status_code = 404


class UploadTooLargeError(ScannerError):
    status_code = 413


class UnsupportedMediaError(ScannerError):
    status_code = 415


class InvalidChecksError(ScannerError):
    status_code = 422


class ModelUnavailableError(ScannerError):
    status_code = 503


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ScannerError)
    async def _scanner_error(_: Request, exc: ScannerError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
