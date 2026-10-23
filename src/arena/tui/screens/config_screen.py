"""Experiment configurator: pick agents/format/augments -> ExperimentConfig YAML -> launch."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static

from arena.config.schema import AgentConfig
from arena.registry.agents import entries as agent_entries
from arena.registry.augments import entries as augment_entries
from arena.tui.config_model import Draft
from arena.tui.widgets.param_form import param_widgets, read_params

_FORMATS = ["gen9randombattle", "gen9ou", "gen9uu", "gen9monotype", "gen9doublesou"]
_AUG_MODES = ["off", "passive", "tool"]
_AUG_LABELS = {
    "damage_calc": "Damage calc", "type_matchup": "Type matchup", "base_stats": "Base stats",
    "ability": "Ability", "pokedex": "Pokedex",
}


class ConfigScreen(Screen):
    TITLE = "Configure"
    BINDINGS = [("q", "quit", "Quit")]
    CSS = """
    ConfigScreen { padding: 1 2; background: $background; }
    #cols { height: 1fr; }
    .col { width: 1fr; height: 1fr; padding: 0 1; }
    #sec_add { height: 1fr; }
    .section {
        border: round $primary 50%;
        background: $surface 0%;
        padding: 0 2;
        margin-bottom: 1;
        height: auto;
        border-title-color: $accent;
        border-title-style: bold;
    }
    .row, .prow { height: 3; }
    .lbl, .pname { width: 18; height: 3; content-align: left middle; color: $text-muted; }
    Select, Input { width: 1fr; background: $surface; }
    Input:focus, Select:focus { border: tall $primary; }
    #agent_params { height: auto; max-height: 10; }
    #roster { height: 1fr; min-height: 6; margin-top: 1; border: round $secondary 60%; border-title-color: $accent; }
    #actions { height: 3; margin-top: 1; }
    #msg { color: $warning; height: 1; }
    Button { margin-right: 2; }
    """

    def __init__(self):
        super().__init__()
        self.draft = Draft()
        self._agents = agent_entries()
        try:
            from arena.llm.model_registry import ModelRegistry
            self._models = ModelRegistry.load().names()
        except Exception:  # noqa: BLE001
            self._models = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="cols"):
            with VerticalScroll(classes="col"):
                with Vertical(classes="section", id="sec_add") as s_add:
                    s_add.border_title = "Add agent"
                    yield Horizontal(
                        Label("Type", classes="lbl"),
                        Select([(e.label, t) for t, e in self._agents.items()],
                               value=next(iter(self._agents)), id="agent_type"),
                        classes="row")
                    yield Horizontal(
                        Label("Name", classes="lbl"),
                        Input(placeholder="display name", id="agent_name"),
                        classes="row")
                    yield VerticalScroll(id="agent_params")
                    yield Button("+ Add agent", id="add_agent", variant="primary")
                    table = DataTable(id="roster")
                    table.border_title = "Roster"
                    table.add_columns("Name", "Type", "Model", "Params")
                    yield table

            with VerticalScroll(classes="col"):
                with Vertical(classes="section") as s_exp:
                    s_exp.border_title = "Experiment"
                    yield Horizontal(Label("Name", classes="lbl"), Input("tui_experiment", id="f_name"), classes="row")
                    yield Horizontal(Label("Format", classes="lbl"),
                                     Select([(f, f) for f in _FORMATS], value="gen9randombattle", id="f_format"),
                                     classes="row")
                    yield Horizontal(Label("Random seed", classes="lbl"), Input("0", id="f_seed"), classes="row")

                with Vertical(classes="section") as s_eval:
                    s_eval.border_title = "Evaluation"
                    yield Horizontal(Label("Match mode", classes="lbl"),
                                     Select([("round_robin", "round_robin"), ("anchored", "anchored")],
                                            value="round_robin", id="f_mode"), classes="row")
                    yield Horizontal(Label("Battles", classes="lbl"), Input("4", id="f_n"), classes="row")
                    yield Horizontal(Label("Anchors", classes="lbl"),
                                     Input(placeholder="comma-separated", id="f_anchors"), classes="row")
                    yield Horizontal(Label("Team mode", classes="lbl"),
                                     Select([("mirror", "mirror"), ("pool_random", "pool_random")],
                                            value="mirror", id="f_teams"), classes="row")
                    yield Horizontal(Label("Cost cap ($)", classes="lbl"),
                                     Input(placeholder="blank = no cap", id="f_cost"), classes="row")

                with Vertical(classes="section") as s_aug:
                    s_aug.border_title = "Augmentations"
                    for aug in augment_entries():
                        yield Horizontal(Label(_AUG_LABELS.get(aug, aug), classes="lbl"),
                                         Select([(m, m) for m in _AUG_MODES], value="off", id=f"aug_{aug}"),
                                         classes="row")

        yield Static("", id="msg")
        yield Horizontal(
            Button("> Start", id="start", variant="success"),
            Button("History", id="history"),
            id="actions")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild_params()

    # --- dynamic param form -------------------------------------------
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "agent_type":
            self._rebuild_params()

    def _rebuild_params(self) -> None:
        box = self.query_one("#agent_params", VerticalScroll)
        box.remove_children()
        entry = self._agents[self._cur_type()]
        widgets = list(param_widgets(entry, self._models))
        if widgets:
            box.mount(*widgets)   # one mount -> single reflow, no per-widget flicker

    def _cur_type(self) -> str:
        v = str(self.query_one("#agent_type", Select).value)
        return v if v in self._agents else next(iter(self._agents))

    # --- actions -------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_agent":
            self._add_agent()
        elif event.button.id == "start":
            self._start()
        elif event.button.id == "history":
            from arena.tui.screens.trajectory_screen import TrajectoryScreen
            self.app.push_screen(TrajectoryScreen())

    def _add_agent(self) -> None:
        entry = self._agents[self._cur_type()]
        box = self.query_one("#agent_params", VerticalScroll)
        try:
            params, model = read_params(box, entry)
        except ValueError as e:
            self._msg(f"Invalid params: {e}")
            return
        if entry.needs_model and not model:
            self._msg(f"{entry.type_id} requires a model"); return
        name = self.query_one("#agent_name", Input).value.strip() or entry.type_id
        if any(a.name == name for a in self.draft.agents):
            self._msg(f"Duplicate name: {name!r}"); return
        self.draft.agents.append(AgentConfig(name=name, type=entry.type_id, model=model, params=params))
        self.query_one("#roster", DataTable).add_row(name, entry.type_id, model or "-", str(params))
        self.query_one("#agent_name", Input).value = ""
        self._msg("")

    def _start(self) -> None:
        d = self.draft
        d.name = self.query_one("#f_name", Input).value.strip() or "tui_experiment"
        d.format = str(self.query_one("#f_format", Select).value)
        d.seed = int(self.query_one("#f_seed", Input).value or 0)
        d.eval_mode = str(self.query_one("#f_mode", Select).value)
        d.n_battles = int(self.query_one("#f_n", Input).value or 4)
        anchors = self.query_one("#f_anchors", Input).value.strip()
        d.anchors = [a.strip() for a in anchors.split(",") if a.strip()]
        d.teams_mode = str(self.query_one("#f_teams", Select).value)
        cost = self.query_one("#f_cost", Input).value.strip()
        d.max_cost_usd = float(cost) if cost else None
        d.augmentations = {aug: str(self.query_one(f"#aug_{aug}", Select).value)
                           for aug in augment_entries()}
        err = d.validate()
        if err:
            self._msg(err); return
        self.app.start_eval(d)

    def _msg(self, text: str) -> None:
        self.query_one("#msg", Static).update(text)
