// Persistent MCTS owns stable node addresses across bounded search slices.
use serde::Deserialize;
use serde_json::{json, Value};
use crate::state::Side;

#[derive(Clone, Deserialize)]
pub struct GoalCondition {
    side: u8,
    pokemon: i32,
    metric: String,
    op: String,
    value: f32,
}

impl GoalCondition {
    fn holds(&self, state: &State) -> bool {
        let side = if self.side == 0 { &state.side_one } else { &state.side_two };
        let mon = if self.pokemon >= 0 { Some(&side.pokemon.pkmn[self.pokemon as usize]) } else { None };
        let active = self.pokemon == side.active_index as i32;
        let value = match self.metric.as_str() {
            "hp" => mon.map(|p| p.hp as f32 / p.maxhp.max(1) as f32),
            "alive" => mon.map(|p| if p.hp > 0 { 1.0 } else { 0.0 }),
            "active" => Some(if active { 1.0 } else { 0.0 }),
            "boost_atk" => Some(if active { side.attack_boost as f32 } else { 0.0 }),
            "boost_spa" => Some(if active { side.special_attack_boost as f32 } else { 0.0 }),
            "boost_spe" => Some(if active { side.speed_boost as f32 } else { 0.0 }),
            "boost_def" => Some(if active { side.defense_boost as f32 } else { 0.0 }),
            "hazards" => Some((side.side_conditions.stealth_rock + side.side_conditions.spikes
                + side.side_conditions.toxic_spikes + side.side_conditions.sticky_web) as f32),
            "pokemon_left" => Some(side.pokemon.pkmn.iter().filter(|p| p.hp > 0
                && p.id.to_string().to_lowercase() != "none").count() as f32),
            "tera_available" => Some(if side.pokemon.pkmn.iter().any(|p| p.terastallized) { 0.0 } else { 1.0 }),
            _ => None,
        };
        value.map(|v| match self.op.as_str() {
            "eq" => v == self.value, "ne" => v != self.value, "le" => v <= self.value,
            "ge" => v >= self.value, "lt" => v < self.value, "gt" => v > self.value, _ => false,
        }).unwrap_or(false)
    }
}

#[derive(Clone, Deserialize)]
pub struct StrategicRoute {
    name: String,
    actor: usize,
    moves: Vec<String>,
    when: Vec<GoalCondition>,
    switch_out: bool,
}
impl StrategicRoute {
    fn matches(&self, choice: &MoveChoice, state: &State) -> bool {
        let side = &state.side_one;
        if side.pokemon.pkmn[self.actor].hp <= 0 { return false; }
        if side.active_index as usize != self.actor {
            return matches!(choice, MoveChoice::Switch(index) if *index as usize == self.actor);
        }
        if self.switch_out && matches!(choice, MoveChoice::Switch(_)) { return true; }
        match choice {
            MoveChoice::Move(_) | MoveChoice::MoveTera(_) => {
                let name = choice.to_string(side).to_lowercase();
                self.moves.iter().any(|m| name == *m || name == format!("{}-tera", m))
            },
            _ => false,
        }
    }
}

#[derive(Clone, Deserialize)]
pub struct StrategicGuidance {
    goal: Vec<GoalCondition>,
    routes: Vec<StrategicRoute>,
}

pub struct PersistentSearch {
    root: Box<Node>,
    children: HashMap<(usize, usize, usize), Box<[Node]>>,
    state: State,
    root_eval: f32,
    rng: StdRng,
    plan: Option<StrategicGuidance>,
    prior: HashMap<String, f32>,
    guided_iterations: u32,
    guided_choices: u32,
    descendant_guided_choices: u32,
    prior_iterations: u32,
    goal_hits: u32,
    max_depth: usize,
    route_counts: HashMap<String, u32>,
    examples: Vec<Value>,
}

// The root and all children are exclusively owned Boxes. Moving this owner never
// moves their pointees; its Python binding serializes access with a Mutex.
unsafe impl Send for PersistentSearch {}

fn strategic_name(choice: &MoveChoice, side: &Side) -> String {
    match choice {
        MoveChoice::Switch(i) => format!("switch_{}", side.pokemon[*i].id.to_string().to_lowercase()),
        _ => format!("move_{}", choice.to_string(side).to_lowercase().replace("-tera", "_tera")),
    }
}

impl PersistentSearch {
    pub fn new(mut state: State, plan: Option<StrategicGuidance>, seed: u64) -> Self {
        let (ours, theirs) = state.root_get_all_options();
        let mut root = Box::new(Node::new());
        unsafe { root.populate(ours, theirs); }
        root.root = true;
        Self { root, children: HashMap::new(), root_eval: evaluate(&state), state,
            rng: StdRng::seed_from_u64(seed), plan, prior: HashMap::new(), guided_iterations: 0,
            guided_choices: 0, descendant_guided_choices: 0, prior_iterations: 0,
            goal_hits: 0, max_depth: 0, route_counts: HashMap::new(), examples: Vec::new() }
    }

    unsafe fn select(&mut self, pointer: *mut Node, depth: usize, guided: bool,
                     path: &mut Vec<String>) -> (*mut Node, usize, usize, bool) {
        self.max_depth = self.max_depth.max(depth);
        let node = &mut *pointer;
        if depth > 0 && guided && self.plan.as_ref().map(|p|
            p.goal.iter().all(|g| g.holds(&self.state))).unwrap_or(false) {
            self.goal_hits += 1;
            return (pointer, 0, 0, true);
        }
        if node.s1_options.is_none() {
            let (ours, theirs) = self.state.get_all_options();
            node.populate(ours, theirs);
        }
        let options = node.s1_options.as_ref().unwrap();
        let route = if guided { self.plan.as_ref().and_then(|p|
            p.routes.iter().find(|r| r.when.iter().all(|c| c.holds(&self.state)))) } else { None };
        let preferred: Vec<usize> = options.iter().enumerate().filter(|(_, o)|
            route.map(|r| r.matches(&o.move_choice, &self.state)).unwrap_or(false)).map(|(i, _)| i).collect();
        let coverage = if node.root { options.iter().enumerate().filter(|(_, o)| o.visits < 32)
            .min_by_key(|(_, o)| o.visits).map(|(i, _)| i) } else { None };
        let mut ours = coverage.unwrap_or(0);
        if coverage.is_none() {
            let mut best = f32::NEG_INFINITY;
            for (i, option) in options.iter().enumerate() {
                if !preferred.is_empty() && !preferred.contains(&i) { continue; }
                let mut score = option.ucb1(node.times_visited);
                if node.root {
                    let name = strategic_name(&option.move_choice, &self.state.side_one);
                    score += 0.4 * self.prior.get(&name).copied().unwrap_or(0.0)
                        * ((node.times_visited + 1) as f32).sqrt() / (option.visits + 1) as f32;
                }
                if score > best { best = score; ours = i; }
            }
        }
        if preferred.contains(&ours) {
            self.guided_choices += 1;
            if depth > 0 { self.descendant_guided_choices += 1; }
            *self.route_counts.entry(route.unwrap().name.clone()).or_insert(0) += 1;
        }
        if guided && path.len() < 10 {
            path.push(strategic_name(&options[ours].move_choice, &self.state.side_one));
        }
        let theirs = node.maximize_ucb_for_side(node.s2_options.as_ref().unwrap());
        let key = (pointer as usize, ours, theirs);
        if let Some(children) = self.children.get_mut(&key) {
            let child = node.sample_node(children as *mut Box<[Node]>, &mut self.rng);
            self.state.apply_instructions(&(*child).instructions.instruction_list);
            self.select(child, depth + 1, guided, path)
        } else {
            (pointer, ours, theirs, false)
        }
    }

    pub fn advance(&mut self, duration_ms: u64, iterations: u32, prior: HashMap<String, f32>) -> String {
        self.prior = prior;
        let started = std::time::Instant::now();
        let before = self.root.times_visited;
        loop {
            let guided = self.plan.is_some() && self.root.times_visited % 4 == 0;
            if guided { self.guided_iterations += 1; }
            if !self.prior.is_empty() { self.prior_iterations += 1; }
            let mut path = Vec::new();
            unsafe {
                let root = &mut *self.root as *mut Node;
                let (node, ours, theirs, done) = self.select(root, 0, guided, &mut path);
                let leaf = if done { node } else {
                    (*node).expand(&mut self.state, ours, theirs, &mut self.children, &mut self.rng)
                };
                if !done && guided && self.plan.as_ref().unwrap().goal.iter().all(|g| g.holds(&self.state)) {
                    self.goal_hits += 1;
                }
                let score = (*leaf).rollout(&mut self.state, &self.root_eval);
                (*leaf).backpropagate(score, &mut self.state);
            }
            if guided && path.len() > 1 && self.examples.len() < 8 {
                self.examples.push(json!(path));
            }
            let done = self.root.times_visited - before;
            if (iterations > 0 && done >= iterations) || (iterations == 0 &&
                done % 16 == 0 && started.elapsed().as_millis() >= duration_ms as u128) { break; }
        }
        self.report(before, started.elapsed().as_secs_f64())
    }

    fn report(&self, before: u32, elapsed: f64) -> String {
        let side = |options: &Vec<MoveNode>, state: &Side| -> serde_json::Map<String, Value> {
            options.iter().map(|o| (strategic_name(&o.move_choice, state),
                json!({"visits": o.visits, "total_score": o.total_score}))).collect()
        };
        json!({"ours": side(self.root.s1_options.as_ref().unwrap(), &self.state.side_one),
               "theirs": side(self.root.s2_options.as_ref().unwrap(), &self.state.side_two),
               "visits": self.root.times_visited, "start_visits": before,
               "nodes": self.children.values().map(|v| v.len()).sum::<usize>() + 1,
               "root_eval": self.root_eval, "slice_wall_s": elapsed,
               "guided_iterations": self.guided_iterations, "guided_choices": self.guided_choices,
               "descendant_guided_choices": self.descendant_guided_choices,
               "prior_iterations": self.prior_iterations, "goal_hits": self.goal_hits,
               "max_depth": self.max_depth, "route_counts": self.route_counts,
               "paths": self.examples}).to_string()
    }

    pub fn root_state(&self) -> String { self.state.serialize() }
}
