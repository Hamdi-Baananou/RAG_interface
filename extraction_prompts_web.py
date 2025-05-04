# extraction_prompts_web.py
# Prompts for extracting data from cleaned web data using detailed definitions

# --- Material Properties ---
MATERIAL_FILLING_WEB_PROMPT = """Material filling describes additives added to the base material in order to influence the mechanical material characteristics. Most common additives are GF (glass-fiber), GB (glass-balls), MF (mineral-fiber) and T (talcum)."""
MATERIAL_NAME_WEB_PROMPT = """Please fill in the material name, which has regarding weight the greatest amount at the whole connector."""

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
NUMBER_OF_ROWS_WEB_PROMPT = """This attribute describes the number of rows, to which the cavities are arranged in a connector. The number of rows is only used for connectors with square or rectangular housing shape and furthermore if all cavities have the same cavity size. The longer side of a connector is the horizontal direction and the number of rows will be count only in the horizontal direction.
If the connector has not straight rows (in horizontal direction) or the cavity sizes are different, the value = 0.
For 1-cavity-connectors, the number of rows is “0”. For connectors with another shape (e.g. round, triangular, etc.) or without straight rows, the number of rows is “0”."""
MECHANICAL_CODING_WEB_PROMPT = """A mechanical coding is designed at the plugged connector and its counterpart. The coding is used to avoid failures during pushing process.
The location of the tongue and groove at the plastic parts are varying with the different mechanical coding (A/B/C/D).
Often the coding is mentioned on the drawing, but sometimes not and then it is only drawn. In this case, we use the value: “no naming”.
If all available coding of a connector family are fitting in a universal coded (= neutral or 0 coding) connector, the universal connector has the coding value = Z.
If the connector has no coding, the value = none."""
COLOUR_WEB_PROMPT = """For assembled parts, the dominant colour of the complete assembly should be filled in.
For a single part connector, the colour of the housing has to be selected.
In case of multi-colour connectors, without a dominant colour, enter the colour value ‘multi'."""
COLOUR_CODING_WEB_PROMPT = """It's the colour used to distinguish between connectors of a connector family, existing in the same drawing.
To be able to talk about colour coding, the following conditions must be met: 1. connectors must have a mechanical coding, 2. connectors must have different/additional color of individual parts in the housing
If the connector has no colour coding -> value = none"""

# --- Sealing & Environmental ---
# Note: Splitting Working Temperature into Max and Min
MAX_WORKING_TEMPERATURE_WEB_PROMPT = """Max. Working Temperature in °C according the drawing/datasheet. If no value is available, please enter the value 999."""
MIN_WORKING_TEMPERATURE_WEB_PROMPT = """Min. Working Temperature in °C according the drawing/datasheet. If no value is available, please enter the value 999."""
HOUSING_SEAL_WEB_PROMPT = """The type of sealing between the connector and its counterpart: Radial Seal / Interface seal."""
WIRE_SEAL_WEB_PROMPT = """Wire seal describes the sealing of the space between wire and cavity wall, when a terminal is fitted in a cavity. There are different possibilities for sealing available: Single wire seal, Injected, Mat seal (includes "gel family seal" and "silicone family seal"), None."""
SEALING_WEB_PROMPT = """Indicates if the connector is: Sealed or Unsealed"""
SEALING_CLASS_WEB_PROMPT = """According to their qualification for usage under different environmental conditions, systems are divided in corresponding protection classes, so-called IP-codes. The abbreviation IP means "International Protection" according DIN; in the English-speaking countries, the classes are called "Ingress Protection". Sealing classes are : IPx0, IPx4, IPx4K, IPx5, IPx6, IPx6K, IPx7, IPx8, IPx9 and IPx9K (IPx1, IPx2, IPx3 are not needed for our business). The IP-codes are fixed in the ISO 20653.
We can use these dependencies ONLY if we have exhausted all possibilities (documents, internet, information form supplier, etc.) to define the IP Class without success : Class S1 (unsealed) is IPx0 / Class S2 (sealed) is IPx7 / Class S3 (sealed - w/high pressure spray) is IPx9K"""

# --- Terminals & Connections ---
CONTACT_SYSTEMS_WEB_PROMPT = """This attribute is used to define which contact system is approved for the connector by the supplier/manufacturer. Is more than one contact system used in a connector, all of them must be filled in this attribute."""
TERMINAL_POSITION_ASSURANCE_WEB_PROMPT = """Indicates the number of available TPAs, which are content of the delivered connector (TPAs preassembled). If a separate TPA or more than one, regularly with their own part number, has to be assembled at LEONI production, the amount is given within HD (Housing Definition). In such cases, then here "0" has to be filled.
To guarantee a further locking of a terminal in a connector - the firstly/primary locking is done by the lances at the terminals or at the housings - a secondary locking is provided, the terminal position assurance = TPA. Sometimes it is named 'Anti-Backout'."""
CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT = """CPA is an additional protection to ensure, that the connector is placed correctly to the counterpart and that the connector won´t be removed unintentional. Sometimes it's named 'Anti-Backout'."""
CLOSED_CAVITIES_WEB_PROMPT = """Here the number of the cavities, which are closed, have to listed. If all cavities are open or the closed cavities haven´t numerations, 'none' has to be entered."""

# --- Assembly & Type ---
PRE_ASSEMBLED_WEB_PROMPT = """This attribute defines if the connector is delivered as an assembly, which has to be disassembled in our production in order to use it.
Connectors with a preassembled TPA and/or CPA and/or lever and/or etc., which haven´t to be disassembled in our production, get the value "No".
If the connector must be disassembled in our production before we can use it, get the value "Yes"."""
CONNECTOR_TYPE_WEB_PROMPT = """The type describes roughly the application area that the connector is designed: Standard, Contact Carrier, Actuator…"""
SET_KIT_WEB_PROMPT = """If a connector is delivered as a 'Set/Kit' with one LEONI part number, means connector with separate accessories (cover, lever, TPA,…) which aren´t preassembled, then it is Yes. All loose pieces are handled with the same Leoni part number.
If all loose pieces have their own LEONI part number, then it is No."""

# --- Specialized Attributes ---
HV_QUALIFIED_WEB_PROMPT = """This attribute is set to "Yes" ONLY when the documentation indicates this property, or the parts are used in an HV-connector or an HV-assembly. Otherwise it´s No. HV is specified as the range greater than 60 V."""
