from pydantic import BaseModel, Field


class TranslatedStory(BaseModel):
    translated_title: str = Field(
        ..., 
        min_length=5, 
        max_length=300,
        description="Suggested translated title of the news story."
    )
    translated_content: str = Field(
        ..., 
        min_length=5,
        description="Translated content of the news story."
    )