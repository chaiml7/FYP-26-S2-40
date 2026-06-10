from .supabase_client import supabase as _supabase
 
def get_client():
    return _supabase
 
__all__ = ["get_client"]
 