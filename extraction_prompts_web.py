# extraction_prompts_web.py
# Prompts for extracting data from cleaned web data (usually key-value pairs)

# --- Material Properties ---
MATERIAL_FILLING_WEB_PROMPT = "Determine the Material filling describes additives added to the base material in order to influence the mechanical material characteristics. Most common additives are GF (glass-fiber), GB (glass-balls), MF (mineral-fiber) and T (talcum)."
MATERIAL_NAME_WEB_PROMPT = "Determine Please fill in the material name, which has regarding weight the greatest amount at the whole connector."

# --- Physical / Mechanical Attributes ---
PULL_TO_SEAT_WEB_PROMPT = "Extract Pull-to-Seat value"
GENDER_WEB_PROMPT = "Extract Gender value"
HEIGHT_MM_WEB_PROMPT = "Extract Height [MM] value"
LENGTH_MM_WEB_PROMPT = "Extract Length [MM] value"
WIDTH_MM_WEB_PROMPT = "Extract Width [MM] value"
NUMBER_OF_CAVITIES_WEB_PROMPT = "Extract Number of Cavities value"
NUMBER_OF_ROWS_WEB_PROMPT = "Extract Number of Rows value"
MECHANICAL_CODING_WEB_PROMPT = """Determine the Mechanical Coding value, A mechanical coding is designed at the plugged connector and its counterpart. The coding is used to avoid failures during pushing process.
The location of the tongue and groove at the plastic parts are varying with the different mechanical coding (A/B/C/D).
Often the coding is mentioned on the drawing, but sometimes not and then it is only drawn. In this case, we use the value: "no naming".
If all available coding of a connector family are fitting in a universal coded (= neutral or 0 coding) connector, the universal connector has the coding value = Z.
If the connector has no coding, the value = none."""
COLOUR_WEB_PROMPT = "Extract Colour value"
COLOUR_CODING_WEB_PROMPT = """Determine Colour Coding using this reasoning chain:

    STEP 1: MECHANICAL CODING PREREQUISITE
    - Confirm existence of mechanical coding:
      * Check for Coding A/B/C/D/Z or physical keying
      - No mechanical coding → Return "none"

    STEP 2: COMPONENT FOCUS IDENTIFICATION
    - Scan primary coding components:
      * CPA latches * TPA inserts * Coding keys
      * Mechanical polarization features
      - Ignore non-coding parts (housing base, seals)

    STEP 3: COLOR DIFFERENTIATION CHECK
    - Compare component colors to base housing:
      * Different color on ≥1 coding component → Proceed
      - Identical colors → Return "none"
    - Validate explicit differentiation purpose:
      * "Color-coded for variant identification"
      * "Visual distinction between versions"

    STEP 4: DOMINANT COLOR SELECTION
    - Hierarchy for color determination:
      1. Explicit coding statements ("Red denotes Type B")
      2. Majority of coding components
      3. Highest contrast vs housing
      4. First mentioned color

    STEP 5: DOCUMENT CONSISTENCY VERIFICATION
    - Require ALL:
      1. Same drawing/family context
      2. Multiple connector variants present
      3. Color-coding purpose clearly stated
    - Reject isolated color mentions

    EXAMPLES:
    "Type A (Blue CPA) vs Type B (Red CPA)"
    → REASONING: [Step1] Mech coding * → [Step3] Color diff * → [Step4] Explicit
    → COLOUR CODING: Blue/Red (depending on variant)

    "Black housing with black CPA/TTA"
    → REASONING: [Step1] Mech coding * → [Step3] No diff → "none"
    → COLOUR CODING: none

    Output format:
    COLOUR CODING: [Color/none/NOT FOUND]
"""

# --- Sealing & Environmental ---
WORKING_TEMPERATURE_WEB_PROMPT = "Extract Working Temperature value(s)"
HOUSING_SEAL_WEB_PROMPT = "Extract Housing Seal value"
WIRE_SEAL_WEB_PROMPT = "Extract Wire Seal value"
SEALING_WEB_PROMPT = "Extract Sealing value"
SEALING_CLASS_WEB_PROMPT = "Extract Sealing Class value (IP Code)"

# --- Terminals & Connections ---
CONTACT_SYSTEMS_WEB_PROMPT = "Extract Contact Systems value(s)"
TERMINAL_POSITION_ASSURANCE_WEB_PROMPT = "Extract Terminal Position Assurance value"
CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT = "Extract Connector Position Assurance value"
CLOSED_CAVITIES_WEB_PROMPT = "Extract Closed Cavities value(s)"

# --- Assembly & Type ---
PRE_ASSEMBLED_WEB_PROMPT = "Extract Pre-Assembled value"
CONNECTOR_TYPE_WEB_PROMPT = "Determine the type of connector. The type describes roughly the application area that the connector is designed: Standard, Contact Carrier, Actuator…"
SET_KIT_WEB_PROMPT = "Extract Set/Kit value"

# --- Specialized Attributes ---
HV_QUALIFIED_WEB_PROMPT = "Extract HV Qualified value"
