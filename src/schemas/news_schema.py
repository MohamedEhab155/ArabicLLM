from pydantic  import BaseModel,Field 
from typing import List , Literal

StoryCategory = Literal[
    "politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science",
    "not_specified"
]


EntityType = Literal[
    "person-male", "person-female", "location", "organization",
    "event", "time", "quantity", "money", "product",
    "law", "disease", "artifact", "not_specified"
]

class Entity(BaseModel):
    entity_value: str = Field(
        ...,
        description="The actual name or value of the entity."
    )
    entity_type: EntityType = Field(
        ...,
        description="The type of recognized entity."
    )

class NewsDetails(BaseModel):

    story_title:str= Field(
        ...,min_length=5,max_length=100,description="A fully informative and SEO optimized title of the story.")

    story_keywords: List[str] = Field(
        ...,
        min_items=1,
        description="Relevant keywords associated with the story."
    )

    story_summary: List[str] = Field(
        ...,
        min_items=1,
        max_items=5,
        description="Summarized key points about the story (1-5 points)."
    )


    story_category: StoryCategory = Field(
        ...,
        description="Category of the news story."
    )


    story_entities: List[Entity] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of identified entities in the story."
    )