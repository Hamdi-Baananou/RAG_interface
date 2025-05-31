# extraction_prompts_web.py
# Prompts for extracting data from cleaned web data using detailed definitions

# --- Material Properties ---
MATERIAL_FILLING_WEB_PROMPT = """Material filling describes additives added to the base material in order to influence the mechanical material characteristics. Most common additives are GF (glass-fiber), GB (glass-balls), MF (mineral-fiber) and T (talcum)."""
MATERIAL_NAME_WEB_PROMPT = """Extract primary polymer material using this reasoning chain:
    STEP 1: MATERIAL IDENTIFICATION
    - Scan for:
      ✓ Explicit polymer declarations (PA66, PBT, etc.)
      ✓ Composite notations (PA6-GF30, PPS-MF15)
      ✓ Additive markers (GF, GB, MF, T)
      ✓ Weight percentages (PA(70%), PBT(30%))

    STEP 2: BASE MATERIAL ISOLATION
    - Remove additives/fillers from composite names:
      PA66-GF30 → PA66
      LCP-MF45 → LCP
    - If additives-only mentioned (GF40):
      → Check context for base polymer
      → Else: NOT FOUND

    STEP 3: WEIGHT HIERARCHY ANALYSIS
    - Compare numerical weights when present:
      PA66(55%)/PA6(45%) → PA66
    - No weights? Use declaration order:
      \"Primary material: PPS, Secondary: LCP\" → PPS

    STEP 4: SPECIFICITY RESOLUTION
    - Prefer exact grades:
      PA66 > PA6 > PA
      PPSU > PPS
    - Handle generics:
      \"Thermoplastic\" + GF → PA
      \"High-temp polymer\" → PPS

    STEP 5: VALIDATION
    - Confirm single material meets ALL:
      1. Base polymer identification
      2. Weight/declaration priority
      3. Specificity requirements
    - Uncertain? → NOT FOUND

    **Examples:**
    - **\"Connector: PA6-GF30 (60% resin)\"**
      → REASONING: [Step1 ✓] PA6+GF → [Step2 ✓] PA6 → [Step3 ✓] 60% → [Step4 ✓] Specific grade → [Step5 ✓] Validated
      → MATERIAL NAME: **PA6**

    - **\"Housing: GF40 Polymer\"**
      → REASONING: [Step1 ✓] GF additive → [Step2 ✗] No base polymer → [Step5 ✗] Uncertain
      → MATERIAL NAME: **NOT FOUND**

    **Output format:**
    MATERIAL NAME: [UPPERCASE]"""

# --- Physical / Mechanical Attributes ---
PULL_TO_SEAT_WEB_PROMPT = """Yes, if the connector is designed to assemble the wires/terminals with pull-to-seat."""
GENDER_WEB_PROMPT = """Male or Female or Unisex (both kind of terminal in the same cavity) or Hybrid (different cavities for both kind of terminals in the same connector)"""
HEIGHT_MM_WEB_PROMPT = """Height is measured in direction Y.
Total height of the connector (in millimeter) according to the supplier drawing. In some rare cases the height is “longer” then the width.
The dimension is measured as if the connector is assembled. When the connector includes a TPA/CPA, it is the dimension in locked position."""
LENGTH_MM_WEB_PROMPT = """Length is measured in direction Z.
Total length of the connector (in millimeter) according to the supplier drawing. Length is measured dimension from mating face (plug-in to counterpart) to back (wire/cable).
The dimension is measured as if the connector is assembled. When the connector includes a TPA/CPA, it is the dimension in locked position."""
WIDTH_MM_WEB_PROMPT = """Width is measured in direction X.
Total width of the connector (in millimeter) according to the supplier drawing. In some rare cases the width is “shorter” then the height.
The dimension is measured as if the connector is assembled. When the connector includes a TPA/CPA, it is the dimension in locked position."""
NUMBER_OF_CAVITIES_WEB_PROMPT = """For connectors the cavities where terminals will be plugged have to be count.
The number of cavities is the highest number that is printed/defined on the housing itself. In most cases, the number of cavities is also noted in the title block (often in a corner) of the drawing."""
NUMBER_OF_ROWS_WEB_PROMPT = """Determine the number of rows"""
MECHANICAL_CODING_WEB_PROMPT = """A mechanical coding is designed at the plugged connector and its counterpart. The coding is used to avoid failures during pushing process.
The location of the tongue and groove at the plastic parts are varying with the different mechanical coding (A/B/C/D).
Often the coding is mentioned on the drawing, but sometimes not and then it is only drawn. In this case, we use the value: “no naming”.
If all available coding of a connector family are fitting in a universal coded (= neutral or 0 coding) connector, the universal connector has the coding value = Z.
If the connector has no coding, the value = none."""
COLOUR_WEB_PROMPT = """For assembled parts, the dominant colour of the complete assembly should be filled in.
For a single part connector, the colour of the housing has to be selected.
In case of multi-colour connectors, without a dominant colour, enter the colour value ‘multi'."""
COLOUR_CODING_WEB_PROMPT = """Determine the Color Coding if found if not its not found"""

# --- Sealing & Environmental ---
# Note: Splitting Working Temperature into Max and Min
MAX_WORKING_TEMPERATURE_WEB_PROMPT = """Max. Working Temperature in °C according the drawing/datasheet. If no value is available, please enter the value 999. max range temperature"""
MIN_WORKING_TEMPERATURE_WEB_PROMPT = """Min. Working Temperature in °C according the drawing/datasheet. If no value is available, please enter the value 999. min range temperature"""
HOUSING_SEAL_WEB_PROMPT = """The type of sealing between the connector and its counterpart: Radial Seal / Interface seal."""
WIRE_SEAL_WEB_PROMPT = """Wire seal describes the sealing of the space between wire and cavity wall, when a terminal is fitted in a cavity. There are different possibilities for sealing available: Single wire seal, Injected, Mat seal (includes "gel family seal" and "silicone family seal"), None."""
SEALING_WEB_PROMPT = """Determine sealing status using this reasoning chain:

    STEP 1: IP CODE EXTRACTION
    - Scan for ISO 20653/IP codes:
      ✓ Valid codes: IPx0, IPx4, IPx4K, IPx5, IPx6, IPx6K, IPx7, IPx8, IPx9, IPx9K
      ✗ Ignore: IPx1, IPx2, IPx3

    STEP 2: IP-BASED CLASSIFICATION
    - If valid IP codes found:
      → IPx0 → **Unsealed**
      → Any other valid code → **Sealed**
    - If multiple IP codes:
      → Use highest protection level (e.g., IPx9K > IPx7)

    STEP 3: FUNCTIONAL SEALING INDICATORS
    - If no valid IP codes:
      ✓ Check for sealing features:
        * \"Waterproof\"/\"dustproof\"
        * \"Sealed\"/\"gasket\"/\"O-ring\"
        * \"Environmental protection\"
      ✓ Check for explicit negatives:
        * \"Unsealed\"/\"no sealing\"

    STEP 4: CONFLICT RESOLUTION
    - Priority hierarchy:
      1. IP codes (STEP 2)
      2. Explicit functional terms (STEP 3)
      3. Default to NOT FOUND

    STEP 5: FINAL VALIDATION
    - **Sealed** requires:
      ✓ IP code ≥IPx4 OR
      ✓ Functional sealing description
    - **Unsealed** requires:
      ✓ IPx0 OR
      ✓ Explicit lack of sealing

    Examples:
    \"IPx9K-rated for high-pressure washdown\"
    → REASONING: [Step1] IPx9K → [Step2] Sealed
    → SEALING: Sealed

    \"No IP rating but includes silicone gasket\"
    → REASONING: [Step1] No IP → [Step3] Gasket → Sealed
    → SEALING: Sealed

    \"IPx0 connector with 'dust-resistant' claim\"
    → REASONING: [Step1] IPx0 → [Step4] Overrides description → Unsealed
    → SEALING: Unsealed

    Output format:
    SEALING: [Sealed/Unsealed/Not Found]"""
SEALING_CLASS_WEB_PROMPT = """Determine the IP sealing class"""

# --- Terminals & Connections ---
CONTACT_SYSTEMS_WEB_PROMPT = """Identify approved contact systems using this reasoning chain:

    STEP 1: SOURCE IDENTIFICATION
    - Scan for:
      ✓ Explicit system families (MQS, MLK, SLK, etc.)
      ✓ Terminal part numbers (123-4567, XW3D-XXXX-XX)
      ✓ Manufacturer approval statements:
        * \"Approved for use with...\"
        * \"Compatible contact systems:\"
        * \"Recommended mating system\"

    STEP 2: MANUFACTURER PRIORITIZATION
    - Verify mentions are supplier-specified:
      ✓ Direct manufacturer recommendations
      ✗ Customer-specific part numbers
      ✗ Generic terminal references

    STEP 3: SYSTEM RESOLUTION HIERARCHY
    1. Primary: Explicit family mentions (MQS 0.64)
    2. Secondary: Part number mapping:
       - Cross-reference with manufacturer catalogs
       - Match patterns (e.g., 928321-1 → TE MCP 1.2)
    3. Reject unidentifiable part numbers

    STEP 4: MULTI-SYSTEM VALIDATION
    - Check for:
      ✓ Multiple approval statements
      ✓ Hybrid connector systems
      ✓ Generation variants (MQS Gen2 vs Gen3)
    - Require explicit documentation for each system

    STEP 5: STANDARDIZATION CHECK
    - Convert to manufacturer nomenclature:
      \"Micro Quadlock\" → MQS
      \"H-MTD\" → HMTD
    - Maintain versioning: MLK 1.2 ≠ MLK 2.0

    Examples:
    \"Approved systems: MQS 0.64 & SLK 2.8 (P/N 345-789)\"
    → REASONING: [Step1] MQS/SLK explicit → [Step2] Approved → [Step5] Standardized
    → CONTACT SYSTEMS: MQS 0.64,SLK 2.8

    \"Terminals: 927356-1 (MCP series)\"
    → REASONING: [Step1] Part number → [Step3] Mapped to MCP → [Step2] Implicit approval
    → CONTACT SYSTEMS: MCP

    \"Compatible with various 2.8mm systems\"
    → REASONING: [Step1] Vague → [Step5] Non-specific → [Final] NOT FOUND
    → CONTACT SYSTEMS: NOT FOUND

    Output format:
    CONTACT SYSTEMS: [system1,system2,.../Not Found]"""
TERMINAL_POSITION_ASSURANCE_WEB_PROMPT = """Indicates the number of available TPAs, which are content of the delivered connector (TPAs preassembled). If a separate TPA or more than one, regularly with their own part number, has to be assembled at LEONI production, the amount is given within HD (Housing Definition). In such cases, then here "0" has to be filled.
To guarantee a further locking of a terminal in a connector - the firstly/primary locking is done by the lances at the terminals or at the housings - a secondary locking is provided, the terminal position assurance = TPA. Sometimes it is named 'Anti-Backout'."""
CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT = """CPA is an additional protection to ensure, that the connector is placed correctly to the counterpart and that the connector won´t be removed unintentional. Sometimes it's named 'Anti-Backout'."""
CLOSED_CAVITIES_WEB_PROMPT = """Here the number of the cavities, which are closed, have to listed. If all cavities are open or the closed cavities haven´t numerations, 'none' has to be entered."""

# --- Assembly & Type ---
PRE_ASSEMBLED_WEB_PROMPT = """This attribute defines if the connector is delivered as an assembly, which has to be disassembled in our production in order to use it.
Connectors with a preassembled TPA and/or CPA and/or lever and/or etc., which haven´t to be disassembled in our production, get the value "No".
If the connector must be disassembled in our production before we can use it, get the value "Yes"."""
CONNECTOR_TYPE_WEB_PROMPT = """Determine the **Type of Connector** using this reasoning chain:

    STEP 1: EXPLICIT TYPE IDENTIFICATION
    - Scan for exact terms:
      ✓ \"Standard\"
      ✓ \"Contact Carrier\"
      ✓ \"Actuator\"
      ✓ Other documented types (e.g., \"Sensor\", \"Power Distribution\")

    STEP 2: CONTEXTUAL INFERENCE
    - If no explicit type:
      ✓ Analyze application context:
        * \"Modular contact housing\" → **Contact Carrier**
        * \"Used in mechanical actuation systems\" → **Actuator**
        * \"General-purpose\" / No special features → **Standard**
      ✓ Map keywords to types:
        * \"Carrier,\" \"module holder\" → Contact Carrier
        * \"Movement,\" \"lever-operated\" → Actuator
        * \"Universal,\" \"base model\" → Standard

    STEP 3: APPLICATION VALIDATION
    - Verify inferred type aligns with:
      ✓ Connector design (e.g., Contact Carriers have modular slots)
      ✓ System integration described (e.g., Actuators link to moving parts)
      ✗ Reject mismatches (e.g., \"Actuator\" term in a static assembly)

    STEP 4: DEFAULT RESOLUTION
    - No explicit/inferred type? → **NOT FOUND**
    - Generic connector without specialized use? → **Standard**

    Examples:
    \"Modular Contact Carrier (P/N CC-234)\"
    → REASONING: [Step1] Explicit → **Contact Carrier**
    → TYPE OF CONNECTOR: Contact Carrier

    \"Connector for actuator assembly in robotic arm\"
    → REASONING: [Step2] \"actuator\" context → **Actuator**
    → TYPE OF CONNECTOR: Actuator

    \"General automotive wiring connector\"
    → REASONING: [Step4] Generic → **Standard**
    → TYPE OF CONNECTOR: Standard

    \"High-voltage junction module\"
    → REASONING: [Step1-2] No matches → [Step4] **NOT FOUND**
    → TYPE OF CONNECTOR: NOT FOUND

    Output format:
    TYPE OF CONNECTOR: [Standard/Contact Carrier/Actuator/Other/Not Found]"""
SET_KIT_WEB_PROMPT = """If a connector is delivered as a 'Set/Kit' with one LEONI part number, means connector with separate accessories (cover, lever, TPA,…) which aren´t preassembled, then it is Yes. All loose pieces are handled with the same Leoni part number.
If all loose pieces have their own LEONI part number, then it is No."""

# --- Specialized Attributes ---
HV_QUALIFIED_WEB_PROMPT = """This attribute is set to "Yes" ONLY when the documentation indicates this property, or the parts are used in an HV-connector or an HV-assembly. Otherwise it´s No. HV is specified as the range greater than 60 V."""