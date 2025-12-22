## Multi-level Data Structure
1. Build a object from a nested json. The json represents a multi-level taxonomy. Expected structure:
        {
            "title1": {
                "sub-title1": {},
                "sub-title2": {}
            },
            "title2": {
                "sub-title1": {},
                "sub-title2": {}
            }
        }

## Multi-level Classification Pipeline
1. Get all items from level 0
2. Use string comparison to select the most relevant item
3. If no child item found, return it
4. If there are child items, repeat from step 1.