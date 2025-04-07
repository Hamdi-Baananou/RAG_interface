# Prompts for material property extraction and other connector attributes

# --- Material Properties ---

MATERIAL_PROMPT = """
Extract material filling additives using this reasoning chain:
    STEP 1: ADDITIVE IDENTIFICATION
    - Scan document sections for:
      ✓ Explicit additive declarations (GF, GB, MF, T)
      ✓ Mechanical property context clues:
        * \"reinforced with...\"
        * \"improved [strength/stiffness] using...\"
        * \"contains X% [additive]\"
      ✓ Negative statements: \"no additives\", \"unfilled\"
    STEP 2: CONTEXT VALIDATION
    - For each candidate additive:
      ✓ Confirm direct mechanical purpose:
        - \"GF30 for stiffness\" → Valid ✓
        - \"GB colorant\" → Rejected ✗ (non-mechanical)
      ✗ Reject incidental mentions:
        - \"MF manufacturing facility\" → Rejected ✗
    STEP 3: NEGATION CHECK
    - If explicit \"no additives\" statement is found:
      ✓ Verify no contradictory mentions.
      ✓ If confirmed, return **\"none\"**.
    STEP 4: STANDARDIZATION
    - Convert equivalents to standard abbreviations:
      * \"Glass fiber\" → GF
      * \"Mineral-filled\" → MF
      * \"Talc-filled\" → T
      * \"Glass beads\" → GB
      * \"Mica-filled\" → MF
    - Reject non-standard or ambiguous terms:
      * \"Carbon additives\" → **NOT FOUND**
    STEP 5: CERTAINTY ASSESSMENT
    - Final check:
      ✓ At least **one** valid additive with mechanical context.
      ✓ No ambiguous or non-standard terms.
      ✓ No conflicting information.
    - If any doubts exist → **NOT FOUND**.
    **Examples:**
    - **PA66-GF30-T15 (improved impact resistance)**
      → REASONING: [Step1] Identified GF, T with mechanical context → [Step2] Valid → [Step4] Standardized
      → MATERIAL FILLING: **GF, T**
    - **Unfilled PPS compound**
      → REASONING: [Step1] \"Unfilled\" found → [Step3] Negative confirmation
      → MATERIAL FILLING: **none**
    - **Contains 5% specialty reinforcement**
      → REASONING: [Step1] Non-standard term → [Step4] Rejected → [Step5] Uncertain
      → MATERIAL FILLING: **NOT FOUND**
    **Output format:**
    REASONING: [Step analysis summary]
    MATERIAL FILLING: [abbreviations/none/NOT FOUND]
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
    REASONING: [Step analysis with ✓/✗ markers]
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
    REASONING: [Key step determinations]
    PULL-TO-SEAT: [Yes/No]
"""