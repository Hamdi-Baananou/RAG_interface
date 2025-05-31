# Prompts for material property extraction and other connector attributes

# --- Material Properties ---

MATERIAL_PROMPT = """
Extract material filling additives:
 Material filling describes additives added to the base material in order to influence the mechanical material characteristics. Most common additives are GF (glass-fiber), GB (glass-balls), MF (mineral-fiber) and T (talcum).
    **Output format:**
    MATERIAL FILLING: [abbreviations/none]
"""

MATERIAL_NAME_PROMPT = """
Extract primary polymer material using this reasoning chain:
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
    MATERIAL NAME: [UPPERCASE]
"""

# --- Physical / Mechanical Attributes ---

PULL_TO_SEAT_PROMPT = """
Determine Pull-To-Seat requirement using this reasoning chain:

    STEP 1: ACTION IDENTIFICATION
    - Scan for:
      ✓ Explicit \"pull-to-seat\" mentions
      ✓ Terminal insertion process descriptions:
        * \"Pull-back assembly required\"
        * \"Tug-lock mechanism\"
        * \"Retract-to-secure\"
      ✓ Alternative methods:
        * \"Pre-inserted terminals\"
        * \"Tool-free insertion\"
        * \"Push-fit design\"

    STEP 2: OPERATIONAL CONTEXT VALIDATION
    - Confirm mentions relate to PRIMARY ASSEMBLY:
      ✓ Terminal/wire installation
      ✓ Final seating action
      ✗ Maintenance/removal procedures
      ✗ Secondary locking mechanisms

    STEP 3: NEGATION HANDLING
    - Check for explicit denials:
      ✓ \"No pull-to-seat required\"
      ✓ \"Self-retaining terminals\"
      ✓ \"Zero-stroke insertion\"
    - Verify no contradictory claims

    STEP 4: ASSEMBLY CONTEXT CONFIRMATION
    - Required pull action must be:
      ✓ Final assembly step
      ✓ Necessary for terminal retention
      ✓ Performed by installer (not tool)
    - If tool-assisted pull: Treat as \"No\"

    STEP 5: FINAL VERIFICATION
    - Meets ALL criteria:
      1. Explicit pull action requirement ✓
      2. Assembly-phase context ✓
      3. No alternative retention methods ✓
    - Any ambiguity → Default to \"No\"

    Examples:
    \"Terminals require pull-back action for seating\"
    → REASONING: [Step1] Pull-back → [Step2] Assembly → [Step4] Manual action
    → PULL-TO-SEAT: Yes

    \"Pre-inserted contacts with CPA secondary lock\"
    → REASONING: [Step1] Pre-inserted → [Step3] Alternative method
    → PULL-TO-SEAT: No

    \"Secure insertion method\"
    → REASONING: [Step1] Vague → [Step5] Ambiguous
    → PULL-TO-SEAT: No

  Output format:
    PULL-TO-SEAT: [Yes/No]
"""

GENDER_PROMPT = """
Determine connector gender using this reasoning chain:

STEP 1: TERMINAL TYPE IDENTIFICATION (Internal Contacts)

    Scan the document for information about the electrical contacts within the housing:
    ✓ Explicit gender terms for contacts: "male pin", "female socket", etc.
    ✓ Physical descriptions of contacts:
    * "Pin contacts", "Tab contacts", "Blade contacts" → Functionally Male Contacts
    * "Socket contacts", "Receptacle contacts" → Functionally Female Contacts
    ✓ Specific terminal part numbers mentioned (requires knowledge base about those terminals, if available).

    Note the functional gender(s) of the internal contacts identified.

STEP 2: CAVITY ARCHITECTURE ANALYSIS (Housing Structure)

    Analyze the housing design based on descriptions or drawings:
    ✓ Number of positions/cavities.
    ✓ If multiple positions, are they intended for the same type of contact (uniform) or different types (mixed)?
    ✓ Explicit mentions of cavity types: "pin cavities", "socket cavities".

    For potential mixed-gender contacts (identified in Step 1):
    ✓ Check cavity configuration:
    * "Same cavity accepts both pin and socket" → Indicates potential Unisex architecture.
    * "Separate, dedicated cavities for pins and sockets" → Indicates Hybrid architecture.

STEP 3: MANUFACTURER NOMENCLATURE (Assembly Identification)

    Interpret the primary name or terminology used by the manufacturer for the overall assembly:

        "Plug", "Header" → Strong indicator of MALE assembly gender.

        "Receptacle", "Socket", "Connector Housing" (if context implies mating to a plug/header) → Strong indicator of FEMALE assembly gender.

        "Combo", "Hybrid Connector" → Likely Hybrid assembly gender.

    Note any gender-specific suffixes in the main assembly part number (less common, e.g., "-M", "-F").

STEP 4: CONFLICT RESOLUTION & GENDER DETERMINATION

    Priority Rule: The Manufacturer Nomenclature (Step 3) for the overall assembly ("Plug" or "Receptacle") is the primary determinant of the final assembly gender classification (Male/Female). This overrides the functional gender of the internal contacts (Step 1) if they conflict.

        Example: A housing named "Plug" (Male indicator) containing internal socket contacts (Female function) is classified as MALE overall.

        Example: A housing named "Receptacle" (Female indicator) containing internal pin contacts (Male function) is classified as FEMALE overall.

    Secondary Checks (apply if Step 3 is ambiguous or indicates Hybrid/Unisex):

        Explicit Gender Declarations: If the document explicitly states "Male connector", "Female connector", "Hybrid", "Unisex", use that declaration, overriding Step 3 if necessary (rare).

        Cavity Configuration Evidence (Step 2): Use this to confirm Hybrid (separate mixed cavities) or Unisex (shared mixed cavities) only if Step 3 indicated such potential or was unclear.

        Internal Contact Types (Step 1): Mainly used to confirm uniformity (all pins or all sockets) or mixed nature for Hybrid/Unisex analysis.

    Final Decision Logic:

        If Step 3 clearly indicates "Plug" (Male) or "Receptacle" (Female) → Assign that gender.

        If Step 3 indicates Hybrid/Combo → Assign Hybrid.

        If Step 2 confirms Unisex architecture → Assign Unisex.

        If unambiguous determination isn't possible following these rules → Assign NOT FOUND.

    Reject unverified assumptions.

STEP 5: FINAL VALIDATION

    Confirm the SINGLE final classification based on the resolution in Step 4:
    ✓ Male: Typically a "Plug" assembly.
    ✓ Female: Typically a "Receptacle" assembly.
    ✓ Hybrid: Contains separate cavities for both Male and Female functional contacts.
    ✓ Unisex: Contains cavities designed to accept both Male and Female functional contacts.

    Ensure the reasoning aligns with the priority rules.

Examples:

"Part Name: Receptacle Assembly. Drawing shows pin contacts in all positions."
→ REASONING: [Step1] Pin contacts (Male function) → [Step3] "Receptacle" = Female assembly → [Step4] Priority Rule applied: Manufacturer Nomenclature 'Receptacle' determines Female assembly gender, overriding internal contact type → [Step5] Uniform Female assembly.
→ GENDER: Female

"Part Name: Plug Assembly. Document specifies applicable socket terminals."
→ REASONING: [Step1] Socket terminals (Female function) → [Step3] "Plug" = Male assembly → [Step4] Priority Rule applied: Manufacturer Nomenclature 'Plug' determines Male assembly gender, overriding internal contact type → [Step5] Uniform Male assembly.
→ GENDER: Male

"Combo Connector: Cavities A1-A5 accept pins, B1-B5 accept sockets."
→ REASONING: [Step1] Both Pin & Socket contacts → [Step2] Separate cavities → [Step3] "Combo" = Likely Hybrid → [Step4] Cavity evidence confirms Hybrid → [Step5] Hybrid assembly.
→ GENDER: Hybrid

Output format:
REASONING: [Key determinations following the steps and priority rule]
GENDER: [Male/Female/Unisex/Hybrid]
"""

HEIGHT_MM_PROMPT = """
Determine connector height using this reasoning chain:

    STEP 1: COORDINATE SYSTEM ANALYSIS
    - Identify connector type:
      ✓ Round: X=Y=diameter → Height = diameter
      ✓ Rectangular: X>Y (unless explicitly overridden)
    - Locate mating face reference in X-Y plane

    STEP 2: COMPONENT ISOLATION
    - Identify height contributors in Y-axis:
      ✓ Base housing
      ✓ CPA/TPA in locked position
      ✓ Seals/gaskets (radial seals excluded)
    - Ignore X-axis protrusions (latches, wire seals)

    STEP 3: MEASUREMENT EXTRACTION
    - Capture values with:
      ✓ Direct Y-axis labels (\"Y=12.5mm\")
      ✓ Implied height terms (\"total height\", \"stack height\")
      ✓ Assembly-specific values (\"when locked\")
    - Reject non-Y dimensions (\"length\", \"width\")

    STEP 4: LOCKED POSITION ADJUSTMENT
    - For CPA/TPA:
      ✓ Add locked position offset if specified
      ✓ Use default engagement values:
        * CPA: +0.8-1.2mm
        * TPA: +0.5-0.7mm
    - Sum sequential engagements:
      \"Primary lock +2mm, secondary +1mm\" → +3mm

    STEP 5: GEOMETRIC CALCULATION
    - Rectangular connectors:
      ✓ Sum component Y-values
      ✓ Maintain axis priority (Y=height even if Y>X)
    - Round connectors:
      ✓ Use diameter directly
      ✓ Ignore component offsets (radial seals)

    STEP 6: VALIDATION
    - Check against physical constraints:
      ✓ Min height = 1.5mm
      ✓ Max height = 150mm
    - Implausible values? → 999

    Examples:
    \"Rectangular housing Y=6.2mm + CPA locked (+1.0mm)\"
    → REASONING: [Step2] Y-components → [Step4] 6.2+1 → [Step5] Sum
    → HEIGHT [MM]: 7.2

    \"Round connector Ø8.4 with radial seal\"
    → REASONING: [Step1] Round → [Step5] Diameter=height
    → HEIGHT [MM]: 8.4

    \"X=15mm/Y=18mm (special profile)\"
    → REASONING: [Step1] Y>X allowed → [Step3] Direct Y-value
    → HEIGHT [MM]: 18

    Output format:
    HEIGHT [MM]: [value/999]
"""

LENGTH_MM_PROMPT = """
Determine connector length using this reasoning chain:

    STEP 1: COORDINATE SYSTEM ANALYSIS
    - Identify connector geometry:
      ✓ Round: X=Y=diameter (Z independent)
      ✓ Rectangular: Z=length from mating face to rear
    - Locate mating face reference in X-Y plane

    STEP 2: COMPONENT ISOLATION
    - Identify Z-axis contributors:
      ✓ Main housing body
      ✓ CPA/TPA in locked position
      ✓ Wire/cable entry features
    - Ignore:
      ✗ X-axis components (latches, seals)
      ✗ Radial seals (perpendicular to Z-axis)

    STEP 3: MEASUREMENT EXTRACTION
    - Capture values with:
      ✓ Explicit Z-axis labels (\"Z=32.5mm\")
      ✓ Length-specific terms:
        * \"Total length from mating face to rear\"
        * \"Insertion depth\"
      ✓ Assembly diagrams with dimension lines

    STEP 4: LOCKED POSITION ADJUSTMENT
    - For CPA/TPA:
      ✓ Add locked position offset if specified
      ✓ Use default engagement values:
        * CPA: +1.0-1.5mm
        * TPA: +0.8-1.2mm
    - Sum sequential engagements:
      \"Primary lock +1.5mm\" → Add to base length

    STEP 5: DOCUMENT PRIORITIZATION
    - For multiple values:
      ✓ Compare document dates
      ✓ Prefer engineering specs over marketing docs
      ✓ Use revision numbers (Rev C > Rev B)

    STEP 6: VALIDATION
    - Check against physical constraints:
      ✓ Min length = 5mm
      ✓ Max length = 500mm
    - Implausible values? → 999

    Examples:
    \"Rectangular Z=25mm + CPA locked (+2.0mm)\"
    → REASONING: [Step2] Z-components → [Step4] 25+2 → [Step6] Valid
    → LENGTH [MM]: 27

    \"Round connector Ø12mm (Z=15mm)\"
    → REASONING: [Step1] Round → [Step3] Direct Z-value
    → LENGTH [MM]: 15

    \"2023 Spec: 40mm | 2025 Spec: 35mm\"
    → REASONING: [Step5] Newer doc → [Step6] Valid
    → LENGTH [MM]: 35

    Output format:
    LENGTH [MM]: [value/999]
"""

WIDTH_MM_PROMPT = """
Determine connector width using this reasoning chain:

    STEP 1: GEOMETRY IDENTIFICATION
    - Classify connector shape:
      ✓ Round: X = Y = diameter
      ✓ Rectangular/other: X = longest axis
    - Validate via:
      ✓ Explicit labels (\"rectangular housing\")
      ✓ Diagram annotations (X-axis callouts)

    STEP 2: COMPONENT ADJUSTMENT
    - For TPA/CPA:
      ✓ Check locked position dimensions
      ✓ Add engagement offsets if specified:
        * Typical CPA: +0.5-1.2mm
        * Typical TPA: +0.3-0.8mm
    - Ignore non-X-axis protrusions (latches, seals)

    STEP 3: MEASUREMENT EXTRACTION
    - Prioritize explicit values:
      ✓ \"X=24.5mm\"
      ✓ \"Width: 18mm (locked)\"
    - For round connectors:
      ✓ Use diameter directly (\"Ø12mm\" → 12)

    STEP 4: DOCUMENT HIERARCHY
    - Resolve conflicts using:
      1. Engineering drawings > Spec sheets
      2. Latest revision > Older versions
      3. Dimensioned diagrams > Text descriptions

    STEP 5: VALIDATION
    - Check physical plausibility:
      ✓ Min width = 2mm
      ✓ Max width = 300mm
    - Implausible values? → NOT FOUND

    Examples:
    \"Rectangular housing X=32mm (CPA locked)\"
    → REASONING: [Step1] Rect ✓ → [Step3] Direct value
    → WIDTH [MM]: 32

    \"Round connector Ø15.5 + TPA (+0.7mm)\"
    → REASONING: [Step1] Round → [Step2] 15.5+0.7 → 16.2
    → WIDTH [MM]: 16.2

    \"X-axis: 25mm (pre-lock) / 26.2mm (CPA engaged)\"
    → REASONING: [Step2] Locked state → 26.2
    → WIDTH [MM]: 26.2

    Output format:
    WIDTH [MM]: [value]
"""

NUMBER_OF_CAVITIES_PROMPT = """
Determine cavity count using this reasoning chain:

    STEP 1: CAVITY IDENTIFICATION
    - Scan document sections for:
      ✓ Title block annotations (1-CAVITY, 3-POSITION)
      ✓ Housing diagrams with position markers
      ✓ Part number suffixes (-2C, -4P, -6W)
      ✓ Technical specifications (\"4-way connector\")

    STEP 2: IRRELEVANT NUMBER FILTERING
    - Explicitly ignore:
      ✗ Temperature classes (C125, T40)
      ✗ Version numbers (Ver 2.1)
      ✗ Unrelated codes (IP67, UL94V-0)
      ✗ Quantity indicators (\"Qty: 50\")

    STEP 3: CONFLICT RESOLUTION
    - For multiple candidates:
      1. Apply document hierarchy:
         • Latest revision date
         • Engineering drawings > Marketing docs
      2. Prefer physical labels over part numbers
      3. Use highest explicit number when ambiguous

    STEP 4: CAVITY SYNONYM RESOLUTION
    - Map terms to numbers:
      ✓ \"Single-cavity\"/\"1-posn\"/\"1-way\" → 1
      ✓ \"Dual\"/\"2P\"/\"Two-position\" → 2
      ✓ \"Triple\"/\"3C\"/\"Three-cav\" → 3
    - Reject non-standard terms (\"Multi-port\")

    STEP 5: FINAL VALIDATION
    - Confirm:
      1. Positive integer (≥1)
      2. From approved sources (title block/diagram/PN)
      3. No conflicting evidence in latest doc

    Examples:
    \"Housing: 4-CAVITY (DW-123 Rev.3)\"
    → REASONING: [Step1] Title block → [Step3] Latest rev → [Step5] Valid
    → NUMBER OF CAVITIES: 4

    \"PN: XT-60-2P (marketing sheet: 3-way)\"
    → REASONING: [Step1] Conflict → [Step3] Prefer PN suffix → [Step4] 2P=2
    → NUMBER OF CAVITIES: 2

    \"Single-row connector (no numbers)\"
    → REASONING: [Step1] No indicators → [Step5] Default
    → NUMBER OF CAVITIES: 999

    Output format:
    NUMBER OF CAVITIES: [value/999]
"""

NUMBER_OF_ROWS_PROMPT = """
Determine the number of rows 
    
"""

MECHANICAL_CODING_PROMPT = """
Determine mechanical coding using this reasoning chain:

    STEP 1: CODING IDENTIFICATION
    - Scan for:
      ✓ Explicit labels: \"Coding A/B/C/D/Z\" (case-sensitive)
      ✓ Diagram markers: Keyed slots/pins without labels
      ✓ Universal coding indicators: \"neutral coding\", \"0-position\"
      ✓ Explicit negatives: \"no mechanical coding\"

    STEP 2: DOCUMENT ANALYSIS
    - For visual-only coding:
      ✓ Check drawing annotations
      ✓ Verify text references:
        → Labeled? Use letter code
        → Unlabeled? → \"no naming\"
    - For family connectors:
      ✓ Cross-reference related parts
      ✓ Confirm universal compatibility

    STEP 3: CODING TYPE RESOLUTION
    1. Explicitly named (A/B/C/D):
       - Verify case match (A≠a)
    2. Universal connector (Z):
       - Requires ALL:
         * Family-wide compatibility
         * Neutral/0-position designation
    3. No coding:
       - Explicit \"none\" statement
       - Absence of physical coding features
    4. Ambiguous:
       - Unlabeled diagram features → \"no naming\"

    STEP 4: CONFLICT RESOLUTION
    - Multiple codings:
      ✓ Apply document hierarchy:
        1. Latest revision date
        2. Engineering drawings > Spec sheets
        3. Part numbers with revision suffixes
      ✓ Reject unversioned conflicts

    STEP 5: VALIDATION CHECK
    - Final requirements:
      1. Case-sensitive exact match
      2. Contextual alignment (mating pair)
      3. Physical feature verification
      4. Single definitive answer

    Examples:
    \"Positioning: Coding C (DWG-123 Rev.2)\"
    → REASONING: [Step1] Explicit → [Step3] Valid case → [Step5] Confirmed
    → MECHANICAL CODING: C

    \"Keyed slots shown in Fig.5 (unlabeled)\"
    → REASONING: [Step2] Visual-only → [Step3] Ambiguous → \"no naming\"
    → MECHANICAL CODING: no naming

    \"Universal connector for all variants\"
    → REASONING: [Step3] Family-wide → Z
    → MECHANICAL CODING: Z

    Output format:
    MECHANICAL CODING: [A/B/C/D/Z/no naming/none]
"""

COLOUR_PROMPT = """
Determine connector color using this reasoning chain:

    STEP 1: PART TYPE CLASSIFICATION
    - Identify from:
      ✓ \"Assembled\"/\"Multi-component\" → Assembly
      ✓ \"Housing\"/\"Single-piece\" → Connector body
      ✓ Component lists with separate colors

    STEP 2: COLOR SOURCE SELECTION
    - For ASSEMBLIES:
      > Analyze all components (housing, latches, seals)
      > Ignore non-structural elements (labels, markings)
    - For SINGLE-PIECE:
      > Focus on housing material description
      > Ignore contact/terminal colors

    STEP 3: DOMINANCE ANALYSIS
    - Evaluate color references:
      ✓ Quantitative: \"70% black cover\"
      ✓ Qualitative: \"primary color\", \"main housing\"
      ✓ Visual hierarchy: \"red base with blue accents\"
    - Multi-color triggers:
      * \"Striped\"/\"checkered\" patterns
      * Equal distribution (\"50-50 split\")
      * Explicit \"multi-color\" statement

    STEP 4: CONTEXT VALIDATION
    - Reject ambiguous terms:
      ✗ \"Natural\" (unless defined)
      ✗ Material-inferred colors (brass=golden)
    - Accept only explicitly stated colors:
      ✓ \"Black housing\"
      ✓ \"Blue CPA latch\"

    STEP 5: FINAL RESOLUTION
    - Apply priority:
      1. Explicit dominant color
      2. Assembly majority
      3. Multi-color indicators
      4. NOT FOUND

    Examples:
    \"Black nylon housing with nickel-plated contacts\"
    → REASONING: [Step1] Single-piece → [Step2] Housing color → [Step4] Explicit
    → COLOUR: Black

    \"Assembly: White cover (60%), grey base (40%)\"
    → REASONING: [Step1] Assembly → [Step3] White dominant → [Step5] Majority
    → COLOUR: White

    \"Red/blue dual-tone design\"
    → REASONING: [Step1] Single-piece → [Step3] Equal prominence → [Step5] Multi
    → COLOUR: multi

    Output format:
    COLOUR: [color/multi]
"""

COLOUR_CODING_PROMPT = """
Determine Colour Coding using this reasoning chain:

    STEP 1: MECHANICAL CODING PREREQUISITE
    - Confirm existence of mechanical coding:
      ✓ Check for Coding A/B/C/D/Z or physical keying
      ✗ No mechanical coding → Return \"none\"

    STEP 2: COMPONENT FOCUS IDENTIFICATION
    - Scan primary coding components:
      ✓ CPA latches ✓ TPA inserts ✓ Coding keys
      ✓ Mechanical polarization features
      ✗ Ignore non-coding parts (housing base, seals)

    STEP 3: COLOR DIFFERENTIATION CHECK
    - Compare component colors to base housing:
      ✓ Different color on ≥1 coding component → Proceed
      ✗ Identical colors → Return \"none\"
    - Validate explicit differentiation purpose:
      * \"Color-coded for variant identification\"
      * \"Visual distinction between versions\"

    STEP 4: DOMINANT COLOR SELECTION
    - Hierarchy for color determination:
      1. Explicit coding statements (\"Red denotes Type B\")
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
    \"Type A (Blue CPA) vs Type B (Red CPA)\"
    → REASONING: [Step1] Mech coding ✓ → [Step3] Color diff ✓ → [Step4] Explicit
    → COLOUR CODING: Blue/Red (depending on variant)

    \"Black housing with black CPA/TTA\"
    → REASONING: [Step1] Mech coding ✓ → [Step3] No diff → \"none\"
    → COLOUR CODING: none

    Output format:
    COLOUR CODING: [Color/none]
"""

# --- Sealing & Environmental ---

WORKING_TEMPERATURE_PROMPT = """
Determine working temperatures using this reasoning chain:

    STEP 1: DATA IDENTIFICATION
    - Scan for:
      ✓ Explicit temperature ranges (e.g., \"-40°C to 125°C\")
      ✓ Discrete values:
        * \"Max. operating temp: 150°C\"
        * \"Minimum: -40°C\"
      ✓ Standard references:
        * UL RTI/IEC/AEC-Q200
        * Automotive/Industrial grades

    STEP 2: MAX TEMPERATURE RESOLUTION
    - Collect all max candidates:
      > Explicit numbers
      > Standard-derived values (e.g., UL RTI 130°C → 130)
    - Apply hierarchy:
      1. Explicit stated maximums
      2. Range upper bounds (\"X°C to Y°C\" → Y)
      3. Standard implications
    - No data? → 999

    STEP 3: MIN TEMPERATURE RESOLUTION
    - Collect min candidates:
      > Explicit numbers
      > Standard implications:
        * Automotive → -40°C
        * Industrial → -20°C
        * MIL-SPEC → -55°C
    - Apply hierarchy:
      1. Explicit stated minimums
      2. Range lower bounds
      3. Standard inferences
    - No data? → 999

    STEP 4: CONFLICT RESOLUTION
    - Validate Max > Min:
      > If explicit conflict (e.g., Max=100, Min=150):
        → Prioritize explicit values
        → Note inconsistency in reasoning
    - Handle multiple standards:
      > Use most stringent applicable

    STEP 5: FINAL VALIDATION
    - Both values = 999? → NOT FOUND
    - Any value ≠999? → Final output

    Examples:
    \"Rated for -40°C → 125°C (AEC-Q200)\"
    → REASONING: [Step1] Explicit range + automotive standard → [Step2] Max=125 → [Step3] Min=-40
    → WORKING TEMPERATURE: 125, -40

    \"Max. temp 150°C (UL RTI)\"
    → REASONING: [Step1] Explicit max + UL → [Step2] Max=150 → [Step3] No min → 999
    → WORKING TEMPERATURE: 150, 999

    \"High-temp polymer connector\"
    → REASONING: [Step1] No data → [Step5] Both 999
    → WORKING TEMPERATURE: NOT FOUND

  Output format:
    WORKING TEMPERATURE: /[Max]/[Min]
"""

HOUSING_SEAL_PROMPT = """
Determine housing seal type using this reasoning chain:

STEP 1: TERM SCAN (case-insensitive)
- Search for matches of:
  ✓ "Radial Seal"
  ✓ "Interface Seal"
- Reject:
  ✗ Plural/gerund forms ("Seals", "Sealing")
  ✗ Combined terms ("Radial/Interface", "Radial or Interface")

STEP 2: SEAL CONTEXT VALIDATION
- Confirm matches refer specifically to:
  ✓ Connector-to-counterpart interface
  ✓ Environmental sealing function
  ✗ Internal terminal seals
  ✗ Secondary locking features

STEP 3: EXCLUSIVE MATCH RESOLUTION
- If multiple matches:
  ✓ Prioritize document hierarchy:
    • "Primary: Radial Seal" → Select Radial
    • "Standard: Interface Seal" → Select Interface
  ✗ Reject ambiguous combinations:
    • "Available with Radial Seal or Interface Seal"

STEP 4: RING SEAL INTERPRETATION
- If "Ring Seal" is found (any case), and:
  ✓ It refers to a connector-to-counterpart sealing function
  → Map it to Radial Seal
- Reasoning:
  • Ring Seals typically function through radial compression
  • Matches Radial Seal's mechanical behavior
- Reject if:
  ✗ Term does not describe housing or counterpart sealing
  ✗ Mention is unrelated (e.g., O-rings in internal terminals)

STEP 5: FINAL VALIDATION
- Strict requirements:
  1. Term match or valid Ring Seal mapping
  2. Valid context
  3. Single validated occurrence
  4. Housing-specific reference

Examples:
"Housing-to-counterpart seal: radial seal"
→ REASONING: [Step1] Match → [Step2] Context → Valid
→ HOUSING SEAL: Radial Seal

"interface seal (P/N RS-456)"
→ REASONING: [Step1] Match → [Step2] Not about housing → Rejected
→ HOUSING SEAL: NOT FOUND

"Radial Seal (primary) + Interface Seal (secondary)"
→ REASONING: [Step1] Multiple → [Step3] Hierarchy → Radial
→ HOUSING SEAL: Radial Seal

"Connector uses a molded ring seal to prevent ingress"
→ REASONING: [Step4] Ring Seal Detected → [Step2] Context → Valid Mapping
→ HOUSING SEAL: Radial Seal

Output format:
REASONING: [Key determinations]
HOUSING SEAL: [Radial Seal / Interface Seal ]

"""

WIRE_SEAL_PROMPT = """
Determine the Wire Seal type:

    Wire seal describes the sealing of the space between wire and cavity wall, when a terminal is fitted in a cavity. There are different possibilities for sealing available: Single wire seal, Injected, Mat seal (includes “gel family seal” and “silicone family seal”), None.

    Output format:
    WIRE SEAL: [Single Wire Seal/Injected/Mat Seal/None]
"""

SEALING_PROMPT = """
Determine sealing status using this reasoning chain:

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
    SEALING: [Sealed/Unsealed]
"""

SEALING_CLASS_PROMPT = """
According to their qualification for usage under different environmental conditions, systems are divided in corresponding protection classes, so-called IP-codes. The abbreviation IP means "International Protection" according DIN; in the English-speaking countries, the classes are called "Ingress Protection".
"""

# --- Terminals & Connections ---

CONTACT_SYSTEMS_PROMPT = """
Identify approved contact systems using this reasoning chain:

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
    CONTACT SYSTEMS: [system1,system2,...]
"""

TERMINAL_POSITION_ASSURANCE_PROMPT = """
Determine Terminal Position Assurance (TPA) count using this reasoning chain:

    STEP 1: TPA IDENTIFICATION
    - Scan documents for:
      ✓ Explicit terms: \"TPA\", \"Terminal Position Assurance\", \"Anti-Backout\"
      ✓ Part numbers with TPA identifiers (e.g., \"-TPA1\", \"-2T\")
      ✓ Assembly diagrams showing TPA components

    STEP 2: PREASSEMBLY STATUS CHECK
    - Verify if TPA is **delivered preinstalled**:
      ✓ \"Preassembled TPA\"
      ✓ \"Included with housing\"
      ✗ \"Assemble TPA during production\" → Return 0

    STEP 3: COUNT RESOLUTION
    - For preassembled TPAs:
      1. Direct count: \"Dual TPAs\" → 2
      2. Position-based inference:
         * \"1 TPA per 12 cavities\" → Total = Cavities ÷ 12
         * Single-position connector → Default 1
    - For multiple TPAs:
      ✓ Sum explicitly stated numbers
      ✓ Reject ambiguous terms (\"Multiple TPAs\" → NOT FOUND)

    STEP 4: VALIDATION
    - Confirm TPA count aligns with:
      ✓ Physical connector size/cavities
      ✓ Manufacturer specifications
    - Implausible counts (e.g., 5 TPAs on 2-cavity connector) → NOT FOUND

    STEP 5: DEFAULT HANDLING
    - No TPA mentions after Step 1? → NOT FOUND
    - Assembly required? → 0

    Examples:
    \"Preassembled dual TPA (P/N TPA2-456)\"
    → REASONING: [Step1] TPA term + P/N → [Step2] Preassembled → [Step3] Explicit count
    → TERMINAL POSITION ASSURANCE: 2

    \"Install TPA-7A during wire harnessing\"
    → REASONING: [Step2] Requires assembly → Return 0
    → TERMINAL POSITION ASSURANCE: 0

    \"6-cavity housing with 1 TPA per 3 cavities\"
    → REASONING: [Step3] 6 ÷ 3 = 2 → Valid
    → TERMINAL POSITION ASSURANCE: 2

    Output format:
    TERMINAL POSITION ASSURANCE: [number/0]
"""

CONNECTOR_POSITION_ASSURANCE_PROMPT = """
Determine Connector Position Assurance (CPA) status using this reasoning chain:

    STEP 1: TERM SCAN
    - Search for explicit terms:
      ✓ **CPA**, **Connector Position Assurance**, **Anti-Backout**
      ✓ Part numbers with CPA identifiers (e.g., \"-CPA\", \"-AB\")
      ✓ Diagram labels: \"CPA Lock\", \"Position Assurance Clip\"

    STEP 2: CONTEXT VALIDATION
    - Confirm terms relate to **connector retention**:
      ✓ \"Prevents unintentional disconnection\"
      ✓ \"Secondary locking mechanism\"
      ✗ Reject unrelated uses (e.g., \"Anti-Backout algorithm\")

    STEP 3: FUNCTIONAL INFERENCE
    - Analyze described/pictured components:
      ✓ Latches, clips, or levers labeled as CPA
      ✓ \"Secure mating\" features requiring deliberate action to disconnect

    STEP 4: CONFLICT RESOLUTION
    - For conflicting mentions (e.g., \"CPA\" vs. \"No secondary lock\"):
      1. Prioritize **latest document version**
      2. Prefer engineering drawings > marketing materials
      3. Use explicit denials (\"No CPA\") over ambiguous terms

    STEP 5: DEFAULT HANDLING
    - No CPA mentions after Steps 1-3? → **NOT FOUND**

    Examples:
    \"Includes CPA latch (P/N CPA-456)\"
    → REASONING: [Step1] Term + P/N → **Yes**
    → CONNECTOR POSITION ASSURANCE: Yes

    \"No secondary locking features\"
    → REASONING: [Step4] Explicit denial → **No**
    → CONNECTOR POSITION ASSURANCE: No

    \"Secure mating interface\"
    → REASONING: [Step1-3] No CPA terms → **NOT FOUND**
    → CONNECTOR POSITION ASSURANCE: NOT FOUND

    Output format:
    CONNECTOR POSITION ASSURANCE: [Yes/No]
"""

CLOSED_CAVITIES_PROMPT = """
Determine closed cavities using this reasoning chain:

    STEP 1: CLOSED CAVITY IDENTIFICATION
    - Scan for:
      ✓ Explicit terms: \"closed cavities\", \"blocked positions\"
      ✓ Numbered lists: \"Closed: 1,3,5\"
      ✓ Diagram annotations: Crossed-out cavity numbers

    STEP 2: NUMERATION VALIDATION
    - Extract **only numbered closed cavities**:
      ✓ Validate numerical sequencing (e.g., 1-5 vs. random)
      ✗ Reject non-numeric descriptors (\"A,B,C closed\")

    STEP 3: OPEN CAVITY CHECK
    - Return `none` if:
      ✓ \"All cavities open\" stated
      ✓ Closed cavities lack numbers (e.g., \"two closed cavities\")
      ✓ Mixed open/closed with no numbered closures

    STEP 4: AMBIGUITY RESOLUTION
    - Return `NOT FOUND` for:
      ✗ Vague terms (\"some closed cavities\")
      ✗ Contradictory statements
      ✗ Missing cavity status

    Examples:
    \"Closed cavities: 2,4,6 (see diagram)\"
    → REASONING: [Step1-2] Explicit numbers → **2,4,6**
    → NAME OF CLOSED CAVITIES: 2,4,6

    \"All cavities open for wire access\"
    → REASONING: [Step3] All open → **none**
    → NAME OF CLOSED CAVITIES: none

    \"Positions 3 and 5 are blocked\"
    → REASONING: [Step1-2] Blocked = closed but unnumbered → **none**
    → NAME OF CLOSED CAVITIES: none

    \"Closed cavities unspecified\"
    → REASONING: [Step4] Ambiguous → **none**
    → NAME OF CLOSED CAVITIES: none

    Output format:
    NAME OF CLOSED CAVITIES: [numbers/none]
"""

# --- Assembly & Type ---

PRE_ASSEMBLED_PROMPT = """
Determine pre-assembly status using this reasoning chain:

    STEP 1: ASSEMBLY IDENTIFICATION
    - Scan for key terms:
      ✓ \"Delivered as an assembly\"
      ✓ \"Requires disassembly in production\"
      ✓ \"Pre-assembled\" (context-dependent)
      ✓ Components: TPA, CPA, lever, etc.

    STEP 2: DISASSEMBLY CONTEXT VALIDATION
    - For \"Yes\":
      ✓ Explicit disassembly requirement for production use
      ✓ Full connector assembly needing breakdown
    - For \"No\":
      ✓ Preassembled components (TPA/CPA/lever) with no disassembly needed
      ✓ Statements like \"ready-to-use assembly\"

    STEP 3: COMPONENT VS FULL ASSEMBLY
    - Differentiate:
      ✓ Full connector assembly → Check for disassembly mandates
      ✓ Individual components → Check if they're add-ons requiring removal

    STEP 4: EXPLICIT STATEMENT PRIORITIZATION
    - Hierarchy of evidence:
      1. Direct disassembly instructions (\"Must disassemble before installation\")
      2. Delivery format (\"Shipped fully assembled\")
      3. Component mentions without disassembly context

    STEP 5: FINAL CLASSIFICATION
    - Return \"Yes\" ONLY if:
      1. Full assembly delivered AND
      2. Explicit disassembly required for production
    - Return \"No\" if:
      1. Components preassembled BUT
      2. No full disassembly needed
    - Default to NOT FOUND otherwise

    Examples:
    \"Fully assembled connector; disassemble terminals before wiring\"
    → REASONING: [Step1] Assembly + disassembly → [Step5] Yes
    → PRE-ASSEMBLED: Yes

    \"Includes preassembled CPA latch (no disassembly required)\"
    → REASONING: [Step2] Component-only → [Step5] No
    → PRE-ASSEMBLED: No

    \"Modular housing with TPA\"
    → REASONING: [Step1] No disassembly context → [Step5] Default
    → PRE-ASSEMBLED: NOT FOUND

    Output format:
    PRE-ASSEMBLED: [Yes/No]
"""

CONNECTOR_TYPE_PROMPT = """
Determine the **Type of Connector** using this reasoning chain:

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
    TYPE OF CONNECTOR: [Standard/Contact Carrier/Actuator/Other]
"""

SET_KIT_PROMPT = """
Determine the **Set/Kit** status using this reasoning chain:

    STEP 1: LEONI PART NUMBER ANALYSIS
    - Extract all LEONI part numbers (e.g., \"L-1234\", \"LEO-5A6B\")
    - If **only one part number** exists:
      ✓ Check if it includes accessories (cover, lever, TPA)
      ✓ Verify accessories lack individual part numbers
    - If **multiple part numbers**:
      ✗ Confirm if they belong to separate components

    STEP 2: ACCESSORY IDENTIFICATION
    - List all included components:
      ✓ \"Cover\", \"lever\", \"TPA\", etc.
    - Validate if accessories are:
      ✓ Documented under the **same part number** → **Yes**
      ✓ Assigned **separate part numbers** → **No**

    STEP 3: PREASSEMBLY CHECK
    - Confirm accessories are **NOT preassembled**:
      ✓ Terms like \"loose pieces\", \"requires assembly\"
      ✗ \"Preinstalled cover\" or \"built-in lever\"

    STEP 4: EXPLICIT STATEMENT PRIORITIZATION
    - Override inferences if:
      ✓ \"Set/Kit\" explicitly stated → **Yes**
      ✓ \"Separate part numbers required\" → **No**

    STEP 5: DEFAULT RESOLUTION
    - Ambiguous part numbers or missing info → **NOT FOUND**

    Examples:
    \"Connector Set (P/N L-789) includes cover, lever (no assembly required)\"
    → REASONING: [Step1] Single P/N + accessories → **Yes**
    → SET/KIT: Yes

    \"Main housing (L-456), Cover (L-457), TPA (L-458)\"
    → REASONING: [Step1] Multiple P/Ns → **No**
    → SET/KIT: No

    \"Kit with unassembled components (P/N L-999)\"
    → REASONING: [Step4] Explicit \"Kit\" → **Yes**
    → SET/KIT: Yes

    \"Connector with accessories (no P/N specified)\"
    → REASONING: [Step5] Ambiguous → **NOT FOUND**
    → SET/KIT: NOT FOUND

    Output format:
    SET/KIT: [Yes/No]
"""

# --- Specialized Attributes ---

HV_QUALIFIED_PROMPT = """
Determine HV qualification using this reasoning chain:

    STEP 1: VOLTAGE ANALYSIS
    - Extract all voltage references:
      ✓ Explicit ranges (\"48-800V\")
      ✓ Nominal values (\"400V system\")
      ✓ Standards (\"IEC 60664-1 Class B\")
    - Immediate disqualifiers:
      ≤60V → Auto \"No\"
      Exactly 60V → \"No\"

    STEP 2: EXPLICIT HV MARKERS
    - Scan for exact terms:
      ✓ \"HV-qualified\"/\"HV-certified\"
      ✓ \"HV-connector\"/\"HV-assembly\"
      ✓ \"High-voltage system/application\"
    - Reject:
      ✗ \"High vibration\"
      ✗ \"High velocity\"

    STEP 3: DOCUMENT HIERARCHY
    - For conflicting claims:
      ✓ Prioritize by:
        1. Certification documents
        2. Technical specifications
        3. Marketing materials
      ✓ Use document dates:
        \"2025 spec overrides 2023\"

    STEP 4: CONTEXTUAL VALIDATION
    - Confirm HV context:
      ✓ Electric vehicles
      ✓ Battery systems >60V
      ✓ Charging infrastructure
    - Reject non-electrical \"HV\":
      ✗ Hydraulic systems
      ✗ HVAC (non-battery)

    STEP 5: FINAL RESOLUTION
    - Require BOTH:
      1. Voltage >60V OR explicit HV term
      2. Explicit HV qualification statement
    - Edge cases:
      ✓ 60.1V without HV term → \"No\"
      ✓ 59V with HV term → \"No\"

    Examples:
    \"800V battery connector (IEC 62196)\"
    → REASONING: [Step1] >60V but no HV term → [Step5] Fails dual req → No
    → HV QUALIFIED: No

    \"HV-qualified per LV215-1\"
    → REASONING: [Step2] Explicit term → [Step5] Valid → Yes
    → HV QUALIFIED: Yes

    \"60V hybrid system with HV markings\"
    → REASONING: [Step1] 60V → Auto reject
    → HV QUALIFIED: No

    Output format:
    HV QUALIFIED: [Yes/No]
"""