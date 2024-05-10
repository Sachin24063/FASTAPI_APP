from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ToolsRoutes

app = FastAPI()
origins = ["*"]

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allowing all HTTP methods
    allow_headers=["*"]    # Allowing all HTTP headers
)

# Define a route for the root endpoint
@app.get("/")
def read_root():
    return "Welcome to My Pinterest Downloader API"

# Include routes from ToolsRoutes
app.include_router(ToolsRoutes.router)
