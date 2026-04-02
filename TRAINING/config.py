# ============================================================
# CONFIG
# ============================================================

LABEL_COLS = [
    "C0", "C1", "DR", "C6", "C7", "C8", "C9",
    "C10", "C11", "C13", "C14", "C15", "C18", "C19", "C22",
    "C25", "C27", "C29", "C32"
]

CLASS_NAMES = {
    "C0": "Normal",
    "C1": "AMD",
    "DR": "DR",
    "C6": "Glaucoma",
    "C7": "Hypertensive_Retinopathy",
    "C8": "Pathological_Myopia",
    "C9": "Tessellated_Fundus",
    "C10": "Vitreous_Degeneration",
    "C11": "BRVO",
    "C13": "Large_Optic_Cup",
    "C14": "Drusen",
    "C15": "Epiretinal_Membrane",
    "C18": "Optic_Disc_Edema",
    "C19": "Myelinated_Nerve_Fibers",
    "C22": "Retinal_Detachment",
    "C25": "Refractive_Media_Opacity",
    "C27": "CSC",
    "C29": "Laser_Spots",
    "C32": "CRVO",
}