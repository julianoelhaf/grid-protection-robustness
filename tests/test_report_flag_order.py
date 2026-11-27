"""Keep archived and corrected measurement-loss flags tied to their channels."""

import pandas as pd
import pytest

from sparse_fault_detect.visualization import create_plots


@pytest.mark.parametrize(
    "flag_order, expected", [(None, (True, False)), ("physical", (False, True))]
)
def test_report_labels_measurement_loss_by_physical_channel(
    tmp_path, monkeypatch, flag_order, expected
):
    monkeypatch.setattr(create_plots, "CSV_DIR", str(tmp_path))
    runs = pd.DataFrame(
        {
            "params.n_samples": ["4000"],
            "params.voltage_loss": ["True"],
            "params.current_loss": ["False"],
            "tags.measurement_loss_flag_order": [flag_order],
        }
    )
    converted = create_plots.convert_datatypes(runs)
    assert tuple(converted.loc[0, ["current_loss", "voltage_loss"]]) == expected


def test_report_rejects_mixed_measurement_loss_flag_orders():
    runs = pd.DataFrame(
        {
            "params.n_samples": ["4000", "4000"],
            "tags.measurement_loss_flag_order": [None, "physical"],
        }
    )
    with pytest.raises(ValueError, match="only legacy runs or only corrected runs"):
        create_plots.convert_datatypes(runs)


def test_report_rejects_duplicate_heatmap_cells():
    params = {
        **create_plots.DEFAULT_PARAMS,
        "fault_target": "fault_target",
        "window_length": 0.05,
        "zeroing_duration_s": 0.045,
    }
    runs = pd.DataFrame([{**params, "mean_f1_score": 0.31}, {**params, "mean_f1_score": 0.99}])
    args = ("window_length", "zeroing_duration_s", "mean_f1_score", "fault_target")
    assert create_plots.create_heatmap_data(runs.iloc[:1], *args).loc[0.05, 0.045] == 0.31
    with pytest.raises(ValueError, match="Multiple finished runs for fault_target"):
        create_plots.create_heatmap_data(runs, *args)


def test_gap_heatmap_uses_milliseconds(monkeypatch):
    plt = create_plots.plt
    output = []

    def capture_ticks(_path, **_kwargs):
        axes = plt.gca()
        output.append(
            (
                axes.get_ylabel(),
                axes.get_yticklabels()[0].get_text(),
                axes.get_xlabel(),
                axes.get_xticklabels()[0].get_text(),
            )
        )

    monkeypatch.setattr(plt, "savefig", capture_ticks)
    combo = create_plots.generate_combinations(
        ["window_length"], ["zeroing_duration_s"], ["mean_f1_score"], ["fault_target"]
    )[0]
    data = pd.DataFrame([[0.314]], index=[0.05], columns=[0.045])
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        create_plots.plot_heatmap(data, combo)
    assert output == [("Window Length (ms)", "50", "Gap duration (ms)", "45")] * 2
    assert data.index[0] == 0.05 and data.columns[0] == 0.045
