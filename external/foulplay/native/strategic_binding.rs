#[pyclass(name = "SearchTree", module = "poke_engine")]
struct PySearchTree {
    inner: std::sync::Mutex<poke_engine::mcts::PersistentSearch>,
}

#[pymethods]
impl PySearchTree {
    #[new]
    fn new(state: String, plan: String, seed: u64) -> PyResult<Self> {
        let guidance = serde_json::from_str(&plan)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: std::sync::Mutex::new(poke_engine::mcts::PersistentSearch::new(
            State::deserialize(&state), guidance, seed)) })
    }

    #[pyo3(signature = (duration_ms, prior, iterations=0))]
    fn advance(&self, py: pyo3::Python<'_>, duration_ms: u64, prior: String, iterations: u32) -> PyResult<String> {
        let distribution = serde_json::from_str(&prior)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(py.detach(|| self.inner.lock().unwrap().advance(duration_ms, iterations, distribution)))
    }

    fn root_state(&self) -> String { self.inner.lock().unwrap().root_state() }
}
