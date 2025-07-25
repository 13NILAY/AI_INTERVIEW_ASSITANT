import axios from '../api/axios';
import useAuth from './useAuth';

const useRefreshToken = () => {
    const {setAuthInfo} = useAuth();


    const refresh = async () => {
        const response = await axios.get('/refresh', { withCredentials: true });
        const accessToken=response?.data?.accessToken;
        const email=response?.data?.email;
        const roles=response?.data?.roles;
        const _id=response?.data?._id;
        setAuthInfo({accessToken},email,roles,_id);
        // setAuth(prev => ({
        //     ...prev,
        //     roles: response.data.roles,
        //     email: response.data.email,
        //     is
        //     accessToken: response.data.accessToken
        // }));
        return response.data.accessToken;
    };

    return refresh;
};

export default useRefreshToken;
