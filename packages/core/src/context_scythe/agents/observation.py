from browsergym.utils.obs import flatten_axtree_to_str


def from_browsergym_dict(
    obs: dict,
    use_axtree=True,
    use_screenshot=False,
):
    """
    Create the observation class from the BrowserGym dict
    """
    obs_dict = dict()
    extra_properties = obs.get("extra_element_properties")
    obs_dict["extra_element_properties"] = extra_properties

    if use_axtree:
        axtree_object = obs.get("axtree_object")
    
        axtree = flatten_axtree_to_str(
            axtree_object,
            extra_properties=extra_properties,
            with_visible=True,
            with_clickable=True,
        )
        viewport_state = obs.get("viewport_state")

        obs_dict["axtree"] = axtree
        obs_dict["viewport_state"] = viewport_state

    active_page_index = obs.get("active_page_index", None)
    # Force cast to int, as it can be a np.array
    active_page_index = [int(i) for i in active_page_index]
    
    open_pages_titles = obs.get("open_pages_titles", None)
    open_pages_urls = obs.get("open_pages_urls", None)

    assert active_page_index is not None
    assert open_pages_titles is not None
    assert open_pages_urls is not None

    obs_dict["active_page_index"] = active_page_index
    obs_dict["open_pages_titles"] = open_pages_titles
    obs_dict["open_pages_urls"] = open_pages_urls

    if use_screenshot:
        raise NotImplementedError("Screenshot support not implemented yet")
    
    last_action_error = obs.get("last_action_error")
    obs_dict["last_action_error"] = last_action_error

    return obs_dict
