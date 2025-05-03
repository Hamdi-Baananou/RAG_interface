# extraction_prompts_web.py
# Prompts for extracting data from cleaned web data (usually key-value pairs)

# --- Material Properties ---
MATERIAL_FILLING_WEB_PROMPT = "Extract Material Filling value"
MATERIAL_NAME_WEB_PROMPT = "Extract Material Name value"

# --- Physical / Mechanical Attributes ---
PULL_TO_SEAT_WEB_PROMPT = "Extract Pull-to-Seat value"
GENDER_WEB_PROMPT = "Extract Gender value"
HEIGHT_MM_WEB_PROMPT = "Extract Height [MM] value"
LENGTH_MM_WEB_PROMPT = "Extract Length [MM] value"
WIDTH_MM_WEB_PROMPT = "Extract Width [MM] value"
NUMBER_OF_CAVITIES_WEB_PROMPT = "Extract Number of Cavities value"
NUMBER_OF_ROWS_WEB_PROMPT = "Extract Number of Rows value"
MECHANICAL_CODING_WEB_PROMPT = "Extract Mechanical Coding value"
COLOUR_WEB_PROMPT = "Extract Colour value"
COLOUR_CODING_WEB_PROMPT = "Extract Colour Coding value"

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
CONNECTOR_TYPE_WEB_PROMPT = "Extract Type of Connector value"
SET_KIT_WEB_PROMPT = "Extract Set/Kit value"

# --- Specialized Attributes ---
HV_QUALIFIED_WEB_PROMPT = "Extract HV Qualified value"
