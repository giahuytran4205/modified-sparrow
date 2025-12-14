use jagua_rs::probs::spp::entities::SPProblem;

pub fn strip_width_is_in_check(prob: &SPProblem) -> bool {
    let diameters_of_all_items = prob
        .instance
        .items
        .iter()
        .map(|(i, q)| i.shape_cd.diameter * *q as f64)
        .sum::<f64>();

    prob.strip_width() < 2.0 * (diameters_of_all_items)
}
