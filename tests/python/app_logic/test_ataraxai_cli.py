from unittest import mock

import ataraxai.app_logic.ataraxai_cli as cli

def test_init_params():
    """
    Test the initialization of the CLI parameters.
    """
    llama_params, llama_generation_params, whisper_params = cli.init_params()
    assert llama_params is not None
    assert llama_params.model_path is not None
    assert llama_generation_params is not None
    assert whisper_params is not None
    assert whisper_params.model is not None
    
    
def test_init_core_ai_service():
    """
    Test the initialization of the CoreAIService with mock parameters.
    """
    llama_params, _, whisper_params = cli.init_params()
    
    with mock.patch('ataraxai.app_logic.ataraxai_cli.core_ai_py.CoreAIService') as MockCoreAIService:
        core_ai_service = cli.initialize_core_ai_service(llama_params, whisper_params)
        assert core_ai_service is not None
        MockCoreAIService.assert_called_once_with()
        core_ai_service.initialize_llama_model.assert_called_once_with(llama_params)
        core_ai_service.initialize_whisper_model.assert_called_once_with(whisper_params)
        
        
