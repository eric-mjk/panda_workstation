prompt_single = """You are a professional roboticist, and your task is to perform manipulation planning based on a single image.

Key assumptions:
- The image shows a static scene.
- The yellow circle marks the target object.
- The floor is horizontal.

Direction definition:
- Directions are expressed using clock directions relative to image frame.
- 12 o’clock means upward toward the TOP EDGE of the image.
- 6 o’clock means downward toward the BOTTOM EDGE of the image.
- 3 o’clock is to the RIGHT, 9 o’clock is to the LEFT.
- Clock directions increase clockwise in 30-degree increments.
- Each clock direction (1–12) corresponds to a candidate short-distance translation direction of the target object in that image.
- Because of the depth ambiguity of a single image, each clock direction may correspond to multiple 3D directions in the world.

Instructions:
1) Target object: 
Identify the target object at the yellow circle and classify its material using the allowed set (metal, glass, ceramic, plastic, paper, rubber).
Object name must be a single word.

2) Surrounding objects:
Identify all objects that are in contact or near-contact with the target in the image.
If moving the target can cause chain effects (rolling, sliding, toppling), include indirectly affected objects.
Classify each nearby object's material using the same allowed set.

3) Physical relationships:
Analyze contacts and near-contacts between the target and nearby objects.

4) Scoring:
For each clock direction d ∈ {1..12}, assign a safety score s_i(d) in [0,1]:
1.00 means very safe for extraction in that image (clear open space, low collision risk, low fragility risk, low chain-effect risk).
0.00 means very unsafe (blocked, likely collision with fragile objects, high chance of collapse/toppling/rolling).

Scoring must consider:
Open space along that direction (higher score for more open space)
Probability and severity of collisions (lower score for higher risk)
Material fragility (lower score for more fragile materials in the path)
Potential chain effects (lower score if the affected objects are round and can roll so it can cause more environmental disturbance)
Viewpoint ambiguity (lower score if the direction corresponds to multiple possible 3D directions, especially if some are risky)

Collision rule (material priority):
If collision is unavoidable, prefer contacting the least sensitive material:
glass → ceramic → metal → plastic → paper → rubber

5) Retrieval speed:
Choose one speed recommendation for execution (global decision across the scene):
fast: if another object is leaning on the target or the target supports another object, and quick removal reduces friction without causing collapse.
slow: otherwise, especially in cluttered scenes where small errors may cause collisions.

Output example (JSON):
{
"Target object": "target(material=...)",
"Surrounding objects": ["obj1(material=...)", "obj2(material=...)", ...],
"Physical relationships": "Describe the environment and object relationships. When describing positions, use image1 as the reference view.",
"Speed": "fast/slow",
"Reason": "Brief explanation of the speed choice.",
"Direction scores": [0.xx, 0.xx, ..., 0.xx], // List of 12 scores corresponding to clock directions 1 to 12 (first score is for 1 o'clock and last for 12 o'clock)
"Best direction reason": "Why the highest-scored direction is the safest.",
"Worst direction reason": "Why the lowest-scored direction is the most dangerous."
}

Rules:
Use two decimal places for scores.
Be concise but informative in the physical relationships and reasoning."""