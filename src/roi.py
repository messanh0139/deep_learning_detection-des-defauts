# Analyse de l'impact ROI (Retour sur Investissement)
# On simule un contexte industriel réaliste pour montrer ce que le modèle
# apporte concrètement en termes de coûts et de gains de productivité

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Le ROI est calculé en comparaison avec l'inspection humaine,
# par rapport à notre système
PARAMS = {
    "pieces_par_heure":            120,   # cadence d'une ligne de contrôle qualité
    "heures_par_jour":              16,   # deux équipes
    "jours_par_an":                250,

    # Coût d'une pièce défectueuse qui passe en production (rebut, retour client)
    "cout_defaut_non_detecte":     150,   # euros

    # Coût d'une fausse alarme (pièce saine rejetée à tort)
    "cout_fausse_alarme":           12,   # euros

    # Taux de défauts réels sur la ligne
    "taux_defaut":                0.05,   # 5 % des pièces

    # Précision typique d'un inspecteur humain sur des défauts de surface
    "precision_humaine":          0.72,  

    # Coût annuel d'un inspecteur visuel (salaire + charges)
    "cout_inspecteur_annuel":    38000,   # euros

    # Nombre d'inspecteurs remplacés / assistés par le système
    "nb_inspecteurs":                2,

    # Coût de déploiement du système (matériel + intégration)
    "cout_deploiement":          35000,   # euros
    "cout_maintenance_annuel":    4000,   # euros
}


def compute_roi(model_accuracy: float, params: dict = PARAMS) -> dict:
    """
    Calcule le ROI en comparant le système IA à l'inspection humaine (baseline).
    Le gain vient de deux sources :
      1. Meilleure détection → moins de défauts qui passent → moins de coûts aval
      2. Réduction des coûts RH d'inspection
    """
    pieces_par_an  = params["pieces_par_heure"] * params["heures_par_jour"] * params["jours_par_an"]
    defauts_par_an = pieces_par_an * params["taux_defaut"]

    precision_ia     = model_accuracy / 100
    precision_humain = params["precision_humaine"]

    # Défauts non détectés par chaque système
    non_detectes_humain = defauts_par_an * (1 - precision_humain)
    non_detectes_ia     = defauts_par_an * (1 - precision_ia)

    # Économies grâce à la meilleure détection (défauts en moins qui coûtent cher)
    economies_detection = (non_detectes_humain - non_detectes_ia) * params["cout_defaut_non_detecte"]

    # Fausses alarmes supplémentaires de l'IA vs humain (estimées à 10% du gain)
    surcout_fausses = defauts_par_an * abs(precision_ia - precision_humain) * 0.1 * params["cout_fausse_alarme"]

    # Économies RH : le système assiste / remplace des inspecteurs
    economies_rh = params["nb_inspecteurs"] * params["cout_inspecteur_annuel"]

    # Coût annuel de fonctionnement du système
    cout_systeme_an = params["cout_maintenance_annuel"]

    # Gain net annuel vs baseline humaine
    gain_annuel = economies_detection + economies_rh - surcout_fausses - cout_systeme_an

    # ROI sur 3 ans après déduction du coût de déploiement initial
    gains_3ans = gain_annuel * 3
    roi_3ans   = (gains_3ans - params["cout_deploiement"]) / params["cout_deploiement"] * 100

    payback_mois = (params["cout_deploiement"] / gain_annuel * 12) if gain_annuel > 0 else float("inf")

    return {
        "pieces_par_an":          pieces_par_an,
        "defauts_par_an":         defauts_par_an,
        "non_detectes_humain":    non_detectes_humain,
        "non_detectes_ia":        non_detectes_ia,
        "economies_detection":    economies_detection,
        "economies_rh":           economies_rh,
        "gain_annuel":            gain_annuel,
        "roi_3ans":               roi_3ans,
        "payback_mois":           payback_mois,
        "model_accuracy":         model_accuracy,
    }


def plot_roi(model_accuracy: float, output_dir: str) -> None:
    metrics = compute_roi(model_accuracy)
    _print_roi_summary(metrics)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Analyse ROI — Système de Détection Automatisée de Défauts\n"
        f"Précision du modèle : {model_accuracy:.1f}%  |  Contexte : Ligne de production acier",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Graphe 1 : ROI cumulé sur 3 ans
    ax1 = fig.add_subplot(gs[0, 0])
    mois  = np.arange(0, 37)
    gains = -PARAMS["cout_deploiement"] + metrics["gain_annuel"] / 12 * mois
    colors = ["#DC2626" if g < 0 else "#16A34A" for g in gains]
    ax1.bar(mois, gains / 1000, color=colors, width=0.8, alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(metrics["payback_mois"], color="#F59E0B", linewidth=2,
                linestyle="--", label=f"Payback : {metrics['payback_mois']:.0f} mois")
    ax1.set_title("ROI cumulé sur 36 mois", fontweight="semibold")
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Gain net cumulé (k€)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Graphe 2 : Comparaison des coûts annuels 
    ax2 = fig.add_subplot(gs[0, 1])
    labels_cout = ["Inspection\nmanuelle", "Système IA\n(total annuel)"]
    cout_manuel = PARAMS["nb_inspecteurs"] * PARAMS["cout_inspecteur_annuel"]
    cout_ia     = PARAMS["cout_maintenance_annuel"] + metrics["non_detectes_ia"] * PARAMS["cout_defaut_non_detecte"]
    valeurs     = [cout_manuel / 1000, cout_ia / 1000]
    bars = ax2.bar(labels_cout, valeurs, color=["#DC2626", "#16A34A"], width=0.5, alpha=0.85)
    ax2.bar_label(bars, fmt="%.1f k€", padding=3, fontsize=10, fontweight="bold")
    ax2.set_title("Coût annuel : Manuel vs IA", fontweight="semibold")
    ax2.set_ylabel("Coût (k€)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(valeurs) * 1.25)

    # Graphe 3 : ROI selon la précision du modèle
    ax3 = fig.add_subplot(gs[1, 0])
    accuracies = np.linspace(70, 100, 100)
    rois       = [compute_roi(a)["roi_3ans"] for a in accuracies]
    ax3.plot(accuracies, rois, color="#2563EB", linewidth=2.5)
    ax3.axvline(model_accuracy, color="#16A34A", linewidth=2, linestyle="--",
                label=f"Notre modèle ({model_accuracy:.1f}%)")
    ax3.axhline(0, color="#DC2626", linewidth=1, linestyle=":")
    ax3.fill_between(accuracies, rois, 0,
                     where=[r > 0 for r in rois], alpha=0.1, color="#16A34A")
    ax3.set_title("ROI 3 ans selon la précision", fontweight="semibold")
    ax3.set_xlabel("Précision du modèle (%)")
    ax3.set_ylabel("ROI 3 ans (%)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Graphe 4 : Gains répartis
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ["Économies RH", "Défauts évités", "Moins de\nfausses alarmes"]
    defauts_evites  = (metrics["non_detectes_humain"] - metrics["non_detectes_ia"]) \
                      * PARAMS["cout_defaut_non_detecte"]
    fausses_evitees = metrics["defauts_par_an"] * 0.1 * 0.6 * PARAMS["cout_fausse_alarme"]
    valeurs_gains = [
        metrics["economies_rh"] / 1000,
        defauts_evites / 1000,
        fausses_evitees / 1000,
    ]
    wedge_colors = ["#2563EB", "#16A34A", "#F59E0B"]
    _, _, autotexts = ax4.pie(
        valeurs_gains, labels=categories, colors=wedge_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax4.set_title("Répartition des gains annuels", fontweight="semibold")

    save_path = os.path.join(output_dir, "roi_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Analyse ROI sauvegardée : {save_path}")


def _print_roi_summary(m: dict) -> None:
    print("\n" + "─" * 62)
    print("  ANALYSE ROI — Impact Métier Estimé")
    print("─" * 62)
    print(f"  Précision du modèle       : {m['model_accuracy']:.1f}%")
    print(f"  Précision inspecteur human: {PARAMS['precision_humaine']*100:.0f}%  (baseline)")
    print(f"  Pièces produites / an     : {m['pieces_par_an']:,.0f}")
    print(f"  Défauts estimés / an      : {m['defauts_par_an']:,.0f}")
    print(f"  Défauts non détectés (IA) : {m['non_detectes_ia']:,.0f}  "
          f"vs {m['non_detectes_humain']:,.0f} (humain)")
    print(f"  Économies sur détection   : {m['economies_detection']:,.0f} €")
    print(f"  Économies RH / an         : {m['economies_rh']:,.0f} €")
    print(f"  Gain net annuel estimé    : {m['gain_annuel']:,.0f} €")
    print(f"  ROI sur 3 ans             : {m['roi_3ans']:.1f}%")
    print(f"  Retour sur investissement : {m['payback_mois']:.0f} mois")
    print("─" * 62 + "\n")

