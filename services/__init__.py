# BioPrint Vision — Services Package
from services.compatibility import get_compatibility_info, COMPATIBILITY_RULES
from services.eligibility import evaluate_donor_eligibility
from services.rarity import evaluate_blood_rarity, RARITY_METRICS
from services.inventory import search_blood_availability, update_inventory_units
from services.reporting import generate_analysis_pdf
from services.assistant import AIAssistant
