from fastapi import FastAPI
from application.src.create_service import load_model
from application.src.predict import router as predict_router

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(title="Churn Prediction API", version="1.0.0")

    # Load the best model (ensures model is available globally)
    app.state.model = load_model()

    # Include prediction routes
    app.include_router(predict_router, prefix="/api", tags=["predictions"])

    # Add health check route
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app
